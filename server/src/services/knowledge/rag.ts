// Answering: retrieve, build a cited prompt within the context budget, call
// the router, parse the [n] citations, and write the provenance record. A
// question with no evidence is refused (policy) rather than guessed.

import crypto from 'crypto';
import type { ChatMessage, TokenUsage } from '@freellmapi/shared/types.js';
import { KnowledgeError } from './config.js';
import { audit, getPolicy } from './governance.js';
import { getOntology } from './ontology.js';
import { knowledgeChat } from './llm.js';
import { retrieve, type RetrievedChunk, type GraphContext } from './retrieval.js';
import { insertQuery, sha256, type BaseRow } from './store.js';
import { estimateTokens } from './chunker.js';

export interface Citation {
  n: number;
  chunkId: number;
  documentId: number;
  title: string;
  source: string;
  ordinal: number;
}

export interface AnswerOptions {
  actor: string;
  kind?: 'query' | 'eval';
  maxChunks?: number;
  model?: string | null;
  /** Enforced system prompt from a client profile, prepended verbatim. */
  systemPrompt?: string | null;
  graph?: boolean;
  temperature?: number;
}

export interface AnswerGovernance {
  policyVersion: string;
  ontologyHash: string;
  redactions: number;
  refused: boolean;
  citationsMissing: boolean;
  embedder: string;
  graphEngine: GraphContext['engine'];
}

export interface KnowledgeAnswer {
  id: string;
  base: { id: number; slug: string; name: string };
  question: string;
  answer: string;
  status: 'ok' | 'refused';
  citations: Citation[];
  sources: RetrievedChunk[];
  graph: GraphContext;
  model: { platform: string; modelId: string } | null;
  usage: TokenUsage | null;
  latencyMs: number;
  governance: AnswerGovernance;
}

export const REFUSAL_TEXT = "I don't have that in the knowledge base.";

function sourceBlock(chunks: RetrievedChunk[]): string {
  return chunks.map((c, i) => {
    const where = c.source ? `${c.documentTitle} — ${c.source}` : c.documentTitle;
    return `[${i + 1}] ${where} (part ${c.ordinal + 1})\n${c.text}`;
  }).join('\n\n');
}

export function buildMessages(baseName: string, question: string, chunks: RetrievedChunk[], systemPrompt: string | null | undefined): ChatMessage[] {
  const system = [
    systemPrompt?.trim() ? `${systemPrompt.trim()}\n` : '',
    `You are Wall-E's knowledge assistant answering from the "${baseName}" knowledge base.`,
    'Answer using ONLY the numbered sources below. Cite the sources that support each claim with their number in square brackets, like [1] or [2][3].',
    `If the sources do not contain the answer, reply exactly: ${REFUSAL_TEXT}`,
    'Be precise and concise. Never invent sources or numbers.',
  ].filter(Boolean).join('\n');
  return [
    { role: 'system', content: system },
    { role: 'user', content: `Sources:\n\n${sourceBlock(chunks)}\n\nQuestion: ${question}` },
  ];
}

/** Keep the highest-ranked chunks that fit the context budget. */
export function fitContext(chunks: RetrievedChunk[], maxTokens: number): RetrievedChunk[] {
  const out: RetrievedChunk[] = [];
  let used = 0;
  for (const c of chunks) {
    const t = estimateTokens(c.text) + 20;
    if (out.length > 0 && used + t > maxTokens) break;
    out.push(c);
    used += t;
    if (used >= maxTokens) break;
  }
  return out;
}

export function parseCitations(answer: string, sources: RetrievedChunk[]): Citation[] {
  const found = new Set<number>();
  for (const m of answer.matchAll(/\[(\d+(?:\s*[,;]\s*\d+)*)\]/g)) {
    for (const part of m[1].split(/[,;]/)) {
      const n = Number(part.trim());
      if (Number.isInteger(n) && n >= 1 && n <= sources.length) found.add(n);
    }
  }
  return [...found].sort((a, b) => a - b).map(n => {
    const s = sources[n - 1];
    return { n, chunkId: s.chunkId, documentId: s.documentId, title: s.documentTitle, source: s.source, ordinal: s.ordinal };
  });
}

export async function answerQuestion(base: BaseRow, question: string, opts: AnswerOptions): Promise<KnowledgeAnswer> {
  const policy = getPolicy();
  const ontology = getOntology();
  const started = Date.now();
  const id = crypto.randomUUID();
  const q = question.trim();
  if (!q) throw new KnowledgeError('question is required');

  const retrieval = await retrieve(base, q, { maxChunks: opts.maxChunks, graph: opts.graph });
  const sources = fitContext(retrieval.chunks, policy.generation.max_context_tokens);
  const retrievalJson = JSON.stringify(sources.map(c => ({ chunkId: c.chunkId, documentId: c.documentId, score: c.score, via: c.via, vectorRank: c.vectorRank, keywordRank: c.keywordRank })));
  const graphJson = JSON.stringify({ engine: retrieval.graph.engine, seeds: retrieval.graph.seeds, nodes: retrieval.graph.nodes, edges: retrieval.graph.edges, timings: retrieval.timings });
  const governance: AnswerGovernance = {
    policyVersion: policy.versionTag, ontologyHash: ontology.hash, redactions: retrieval.redactions, refused: false, citationsMissing: false,
    embedder: retrieval.embedder.model, graphEngine: retrieval.graph.engine,
  };
  const baseRef = { id: base.id, slug: base.slug, name: base.name };
  const kind = opts.kind ?? 'query';

  if (sources.length === 0 && policy.generation.refuse_without_evidence) {
    governance.refused = true;
    const latencyMs = Date.now() - started;
    insertQuery({
      id, base_id: base.id, kind, actor: opts.actor, question: q, question_hash: sha256(q), retrieval_json: retrievalJson, graph_json: graphJson,
      prompt_hash: null, platform: null, model_id: null, answer: REFUSAL_TEXT, answer_hash: sha256(REFUSAL_TEXT), citations_json: '[]', usage_json: '{}',
      governance_json: JSON.stringify(governance), status: 'refused', error: null, latency_ms: latencyMs,
    });
    audit({ action: 'query.refused', actor: opts.actor, baseId: base.id, target: `query:${id}`, details: { reason: 'no_evidence' } }, policy);
    return { id, base: baseRef, question: q, answer: REFUSAL_TEXT, status: 'refused', citations: [], sources: [], graph: retrieval.graph, model: null, usage: null, latencyMs, governance };
  }

  const messages = buildMessages(base.name, retrieval.redactedQuestion, sources, opts.systemPrompt);
  const promptHash = sha256(JSON.stringify(messages));
  try {
    const reply = await knowledgeChat(messages, {
      temperature: opts.temperature ?? policy.generation.temperature, maxTokens: 1024, model: opts.model ?? null,
      // Evals are batch work: always wait out a reset. Interactive queries follow the policy.
      waitForResetSeconds: kind === 'eval' ? Math.max(180, policy.generation.wait_for_reset_seconds) : policy.generation.wait_for_reset_seconds,
    });
    const answer = reply.text.trim();
    const isRefusal = answer.includes(REFUSAL_TEXT);
    const citations = isRefusal ? [] : parseCitations(answer, sources);
    governance.citationsMissing = policy.generation.require_citations && !isRefusal && citations.length === 0;
    governance.refused = isRefusal;
    const latencyMs = Date.now() - started;
    insertQuery({
      id, base_id: base.id, kind, actor: opts.actor, question: q, question_hash: sha256(q), retrieval_json: retrievalJson, graph_json: graphJson,
      prompt_hash: promptHash, platform: reply.platform, model_id: reply.modelId, answer, answer_hash: sha256(answer),
      citations_json: JSON.stringify(citations), usage_json: JSON.stringify(reply.usage ?? {}), governance_json: JSON.stringify(governance),
      status: isRefusal ? 'refused' : 'ok', error: null, latency_ms: latencyMs,
    });
    audit({ action: isRefusal ? 'query.refused' : 'query', actor: opts.actor, baseId: base.id, target: `query:${id}`, details: { model: `${reply.platform}/${reply.modelId}`, sources: sources.length, citations: citations.length, citationsMissing: governance.citationsMissing } }, policy);
    return { id, base: baseRef, question: q, answer, status: isRefusal ? 'refused' : 'ok', citations, sources, graph: retrieval.graph, model: { platform: reply.platform, modelId: reply.modelId }, usage: reply.usage, latencyMs, governance };
  } catch (err: any) {
    const message = String(err?.message ?? err).slice(0, 300);
    insertQuery({
      id, base_id: base.id, kind, actor: opts.actor, question: q, question_hash: sha256(q), retrieval_json: retrievalJson, graph_json: graphJson,
      prompt_hash: promptHash, platform: null, model_id: null, answer: null, answer_hash: null, citations_json: '[]', usage_json: '{}',
      governance_json: JSON.stringify(governance), status: 'error', error: message, latency_ms: Date.now() - started,
    });
    audit({ action: 'query.error', actor: opts.actor, baseId: base.id, target: `query:${id}`, details: { error: message } }, policy);
    // Retrieval already happened; let callers (evals) score it even though
    // generation failed.
    if (err && typeof err === 'object') (err as { knowledgeSources?: RetrievedChunk[] }).knowledgeSources = sources;
    throw err;
  }
}
