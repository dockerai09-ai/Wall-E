// Provenance: reassemble a query's lineage (question -> retrieval -> graph ->
// prompt -> model -> answer -> citations) from the knowledge_queries row plus
// the CURRENT state of the documents it used, and export it as W3C PROV-JSON.

import { getQuery, getChunks, getDocument, getBase, type QueryRow } from './store.js';

export interface ProvenanceSource {
  chunkId: number;
  documentId: number;
  score: number;
  via: string[];
  vectorRank: number | null;
  keywordRank: number | null;
  document: { title: string; source: string; status: string; contentHash: string; deletedAt: string | null } | null;
  chunk: { ordinal: number; contentHash: string; text: string } | null;
}

export interface ProvenanceRecord {
  id: string;
  base: { id: number; slug: string; name: string } | null;
  kind: string;
  actor: string;
  createdAt: string;
  status: QueryRow['status'];
  error: string | null;
  latencyMs: number;
  question: string;
  questionHash: string;
  promptHash: string | null;
  model: { platform: string; modelId: string } | null;
  answer: string | null;
  answerHash: string | null;
  usage: Record<string, unknown>;
  citations: unknown[];
  retrieval: ProvenanceSource[];
  graph: Record<string, unknown>;
  governance: Record<string, unknown>;
}

function parse<T>(json: string, fallback: T): T {
  try { return JSON.parse(json) as T; } catch { return fallback; }
}

export function getProvenance(id: string): ProvenanceRecord | undefined {
  const row = getQuery(id);
  if (!row) return undefined;
  const base = getBase(row.base_id);
  const retrieved = parse<{ chunkId: number; documentId: number; score: number; via: string[]; vectorRank: number | null; keywordRank: number | null }[]>(row.retrieval_json, []);
  const chunks = new Map(getChunks(retrieved.map(r => r.chunkId)).map(c => [c.id, c]));
  const docs = new Map<number, ReturnType<typeof getDocument>>();
  const retrieval: ProvenanceSource[] = retrieved.map(r => {
    if (!docs.has(r.documentId)) docs.set(r.documentId, getDocument(r.documentId));
    const d = docs.get(r.documentId);
    const c = chunks.get(r.chunkId);
    return {
      ...r,
      document: d ? { title: d.title, source: d.source, status: d.status, contentHash: d.content_hash, deletedAt: d.deleted_at_ms ? new Date(d.deleted_at_ms).toISOString() : null } : null,
      chunk: c ? { ordinal: c.ordinal, contentHash: c.content_hash, text: c.text } : null,
    };
  });
  return {
    id: row.id,
    base: base ? { id: base.id, slug: base.slug, name: base.name } : null,
    kind: row.kind,
    actor: row.actor,
    createdAt: new Date(row.created_at_ms).toISOString(),
    status: row.status,
    error: row.error,
    latencyMs: row.latency_ms,
    question: row.question,
    questionHash: row.question_hash,
    promptHash: row.prompt_hash,
    model: row.platform && row.model_id ? { platform: row.platform, modelId: row.model_id } : null,
    answer: row.answer,
    answerHash: row.answer_hash,
    usage: parse(row.usage_json, {}),
    citations: parse(row.citations_json, []),
    retrieval,
    graph: parse(row.graph_json, {}),
    governance: parse(row.governance_json, {}),
  };
}

/** W3C PROV-JSON (https://www.w3.org/Submission/prov-json/) view of a record. */
export function toProvJson(rec: ProvenanceRecord): Record<string, unknown> {
  const prefix = { walle: 'urn:walle:knowledge:', prov: 'http://www.w3.org/ns/prov#' };
  const entity: Record<string, unknown> = {};
  const activity: Record<string, unknown> = {};
  const agent: Record<string, unknown> = {};
  const used: Record<string, unknown> = {};
  const wasGeneratedBy: Record<string, unknown> = {};
  const wasDerivedFrom: Record<string, unknown> = {};
  const wasAttributedTo: Record<string, unknown> = {};
  const wasAssociatedWith: Record<string, unknown> = {};
  let n = 0;
  const next = (kind: string) => `_:${kind}${++n}`;

  const qId = `walle:question/${rec.id}`;
  entity[qId] = { 'prov:type': 'Question', 'walle:hash': rec.questionHash, 'walle:text': rec.question };
  const retrievalId = `walle:retrieval/${rec.id}`;
  activity[retrievalId] = { 'prov:type': 'Retrieval', 'prov:startTime': rec.createdAt, 'walle:graphEngine': (rec.graph as { engine?: string }).engine ?? 'none' };
  used[next('u')] = { 'prov:activity': retrievalId, 'prov:entity': qId };
  agent['walle:actor/' + encodeURIComponent(rec.actor || 'anonymous')] = { 'prov:type': 'prov:Person', 'walle:actor': rec.actor };
  wasAssociatedWith[next('a')] = { 'prov:activity': retrievalId, 'prov:agent': 'walle:actor/' + encodeURIComponent(rec.actor || 'anonymous') };

  for (const s of rec.retrieval) {
    const docId = `walle:document/${s.documentId}`;
    if (!entity[docId]) entity[docId] = { 'prov:type': 'Document', 'walle:title': s.document?.title ?? null, 'walle:source': s.document?.source ?? null, 'walle:contentHash': s.document?.contentHash ?? null, 'walle:status': s.document?.status ?? 'unknown' };
    const chunkId = `walle:chunk/${s.chunkId}`;
    entity[chunkId] = { 'prov:type': 'Chunk', 'walle:ordinal': s.chunk?.ordinal ?? null, 'walle:contentHash': s.chunk?.contentHash ?? null, 'walle:score': s.score, 'walle:via': s.via.join(',') };
    wasDerivedFrom[next('d')] = { 'prov:generatedEntity': chunkId, 'prov:usedEntity': docId };
    used[next('u')] = { 'prov:activity': retrievalId, 'prov:entity': chunkId };
  }

  if (rec.answer != null) {
    const genId = `walle:generation/${rec.id}`;
    activity[genId] = { 'prov:type': 'Generation', 'walle:promptHash': rec.promptHash, 'walle:status': rec.status };
    const ansId = `walle:answer/${rec.id}`;
    entity[ansId] = { 'prov:type': 'Answer', 'walle:hash': rec.answerHash, 'walle:text': rec.answer };
    wasGeneratedBy[next('g')] = { 'prov:entity': ansId, 'prov:activity': genId, 'prov:time': rec.createdAt };
    used[next('u')] = { 'prov:activity': genId, 'prov:entity': qId };
    if (rec.model) {
      const modelId = `walle:model/${rec.model.platform}/${rec.model.modelId}`;
      agent[modelId] = { 'prov:type': 'prov:SoftwareAgent', 'walle:platform': rec.model.platform, 'walle:model': rec.model.modelId };
      wasAssociatedWith[next('a')] = { 'prov:activity': genId, 'prov:agent': modelId };
      wasAttributedTo[next('t')] = { 'prov:entity': ansId, 'prov:agent': modelId };
    }
    for (const c of rec.citations as { chunkId: number }[]) {
      wasDerivedFrom[next('d')] = { 'prov:generatedEntity': ansId, 'prov:usedEntity': `walle:chunk/${c.chunkId}`, 'prov:type': 'Citation' };
    }
    const gov = rec.governance as { policyVersion?: string; ontologyHash?: string };
    if (gov.policyVersion) {
      const polId = `walle:policy/${gov.policyVersion}`;
      entity[polId] = { 'prov:type': 'GovernancePolicy', 'walle:ontologyHash': gov.ontologyHash ?? null };
      used[next('u')] = { 'prov:activity': genId, 'prov:entity': polId };
      used[next('u')] = { 'prov:activity': retrievalId, 'prov:entity': polId };
    }
  }

  return { prefix, entity, activity, agent, used, wasGeneratedBy, wasDerivedFrom, wasAttributedTo, wasAssociatedWith };
}
