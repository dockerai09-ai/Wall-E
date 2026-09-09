// Knowledge-graph extraction: one model call per chunk, constrained to the
// ontology, validated, merged into SQLite (entities by class + canonical
// name, one relation row per evidencing chunk) and mirrored to Neo4j.

import type { ChatMessage } from '@freellmapi/shared/types.js';
import { z } from 'zod';
import { getOntology, isValidClass, isValidRelation, ontologyPromptBlock, type Ontology } from './ontology.js';
import { getPolicy, type Policy } from './governance.js';
import { knowledgeChat, extractJsonObject } from './llm.js';
import { addMention, addRelation, canonicalName, upsertEntity, setChunkGraphStatus, getDocument, type BaseRow, type ChunkRow, type EntityRow } from './store.js';
import { mirrorChunkGraph } from './neo4j.js';

const extractedSchema = z.object({
  entities: z.array(z.object({
    name: z.string().min(1).max(200),
    class: z.string().min(1).max(64),
    properties: z.record(z.unknown()).optional(),
    confidence: z.number().min(0).max(1).optional(),
  })).default([]),
  relations: z.array(z.object({
    from: z.string().min(1).max(200),
    to: z.string().min(1).max(200),
    type: z.string().min(1).max(64),
    confidence: z.number().min(0).max(1).optional(),
  })).default([]),
});

export interface ExtractedEntity { name: string; class: string; properties: Record<string, unknown>; confidence: number }
export interface ExtractedRelation { from: string; to: string; type: string; confidence: number }
export interface ExtractedGraph {
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
  dropped: { entities: number; relations: number };
  raw: string;
  platform: string;
  modelId: string;
}

export function extractionMessages(text: string, ontology: Ontology): ChatMessage[] {
  const system = [
    'You are an information-extraction system building a knowledge graph.',
    'Extract the entities and the relations between them that the text explicitly supports.',
    'Use ONLY the entity classes and relation types listed below. Anything else is discarded.',
    '',
    ontologyPromptBlock(ontology),
    '',
    'Rules:',
    '- Use canonical, complete names ("Neo4j", not "the database"); merge duplicates.',
    `- At most ${ontology.extraction.max_entities_per_chunk} entities and ${ontology.extraction.max_relations_per_chunk} relations.`,
    '- confidence is 0..1; be honest about weak evidence.',
    '- Relation "from" and "to" must be names from your entities list.',
    '- Return ONLY a JSON object, no prose, in exactly this shape:',
    '{"entities":[{"name":"...","class":"Person","properties":{},"confidence":0.9}],"relations":[{"from":"...","to":"...","type":"WORKS_FOR","confidence":0.8}]}',
    '- If nothing qualifies return {"entities":[],"relations":[]}.',
  ].join('\n');
  return [
    { role: 'system', content: system },
    { role: 'user', content: `Text:\n"""\n${text}\n"""` },
  ];
}

/** Validate a parsed model reply against the ontology and policy thresholds. */
export function validateExtraction(raw: unknown, ontology: Ontology, policy: Policy): Omit<ExtractedGraph, 'raw' | 'platform' | 'modelId'> {
  const parsed = extractedSchema.safeParse(raw);
  if (!parsed.success) return { entities: [], relations: [], dropped: { entities: 0, relations: 0 } };
  const minConf = Math.max(policy.graph.min_confidence, ontology.extraction.min_confidence);
  let droppedEntities = 0;
  let droppedRelations = 0;
  const entities: ExtractedEntity[] = [];
  const seen = new Map<string, ExtractedEntity>();
  for (const e of parsed.data.entities) {
    const confidence = e.confidence ?? 0.7;
    const name = e.name.trim();
    if (!name || !isValidClass(ontology, e.class) || confidence < minConf) { droppedEntities++; continue; }
    const key = `${e.class}:${canonicalName(name)}`;
    if (seen.has(key)) continue;
    const ent = { name, class: e.class, properties: e.properties ?? {}, confidence };
    seen.set(key, ent);
    entities.push(ent);
    if (entities.length >= ontology.extraction.max_entities_per_chunk) break;
  }
  const byCanonical = new Map<string, ExtractedEntity>();
  for (const e of entities) byCanonical.set(canonicalName(e.name), e);
  const relations: ExtractedRelation[] = [];
  const seenRel = new Set<string>();
  for (const r of parsed.data.relations) {
    const confidence = r.confidence ?? 0.7;
    const from = byCanonical.get(canonicalName(r.from));
    const to = byCanonical.get(canonicalName(r.to));
    if (!from || !to || from === to || confidence < minConf || !isValidRelation(ontology, r.type, from.class, to.class)) { droppedRelations++; continue; }
    const key = `${canonicalName(from.name)}|${r.type}|${canonicalName(to.name)}`;
    if (seenRel.has(key)) continue;
    seenRel.add(key);
    relations.push({ from: from.name, to: to.name, type: r.type, confidence });
    if (relations.length >= ontology.extraction.max_relations_per_chunk) break;
  }
  return { entities, relations, dropped: { entities: droppedEntities, relations: droppedRelations } };
}

export async function extractGraphFromText(text: string, ontology = getOntology(), policy = getPolicy()): Promise<ExtractedGraph> {
  const reply = await knowledgeChat(extractionMessages(text, ontology), {
    temperature: 0,
    maxTokens: 1500,
    model: policy.graph.extraction_model,
    // Background work can afford to wait out a rate-limit window.
    waitForResetSeconds: 120,
  });
  const parsed = extractJsonObject(reply.text);
  const validated = validateExtraction(parsed ?? {}, ontology, policy);
  return { ...validated, raw: reply.text, platform: reply.platform, modelId: reply.modelId };
}

export interface ChunkGraphResult {
  status: 'done' | 'error';
  entities: EntityRow[];
  relationsAdded: number;
  dropped: { entities: number; relations: number };
  mirrored: boolean;
  error?: string;
}

/** Extract, store and mirror the graph for one chunk; updates graph_status. */
export async function processChunkGraph(base: BaseRow, chunk: ChunkRow, ontology = getOntology(), policy = getPolicy()): Promise<ChunkGraphResult> {
  try {
    const extracted = await extractGraphFromText(chunk.text, ontology, policy);
    const rows = new Map<string, EntityRow>();
    const mentions: { entityId: number; confidence: number }[] = [];
    for (const e of extracted.entities) {
      const row = upsertEntity(base.id, e.class, e.name, e.properties);
      rows.set(canonicalName(e.name), row);
      addMention(chunk.id, row.id, e.confidence);
      mentions.push({ entityId: row.id, confidence: e.confidence });
    }
    let relationsAdded = 0;
    const relations: { fromId: number; toId: number; type: string; confidence: number }[] = [];
    for (const r of extracted.relations) {
      const from = rows.get(canonicalName(r.from));
      const to = rows.get(canonicalName(r.to));
      if (!from || !to) continue;
      if (addRelation(base.id, from.id, to.id, r.type, r.confidence, chunk.id)) relationsAdded++;
      relations.push({ fromId: from.id, toId: to.id, type: r.type, confidence: r.confidence });
    }
    setChunkGraphStatus(chunk.id, 'done');
    let mirrored = false;
    const document = getDocument(chunk.document_id);
    if (document) {
      try {
        mirrored = await mirrorChunkGraph({ base, document, chunk, entities: [...rows.values()], mentions, relations });
      } catch (err: any) {
        console.warn(`[knowledge/neo4j] mirror failed for chunk ${chunk.id}: ${String(err?.message ?? err).slice(0, 200)}`);
      }
    }
    return { status: 'done', entities: [...rows.values()], relationsAdded, dropped: extracted.dropped, mirrored };
  } catch (err: any) {
    setChunkGraphStatus(chunk.id, 'error');
    return { status: 'error', entities: [], relationsAdded: 0, dropped: { entities: 0, relations: 0 }, mirrored: false, error: String(err?.message ?? err).slice(0, 300) };
  }
}
