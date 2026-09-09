// Retrieval: BM25 over the FTS5 index and cosine over the embeddings, fused
// with reciprocal-rank fusion, then expanded through the knowledge graph
// (entity linking on the question -> k-hop neighbourhood -> evidence chunks).
// Scores are normalised so the best chunk is 1.0 and governance min_score is
// a fraction of that. Every step's contribution is recorded on the hit
// (`via`, ranks) so provenance can explain why a chunk was used.

import { getDb } from '../../db/index.js';
import { getPolicy, redactText } from './governance.js';
import { embedderFor } from './embedder.js';
import { normalize, vectorSearch } from './vector.js';
import {
  hasFts, getChunks, chunksMentioning, traverse, getDocument, getEntity,
  type BaseRow, type DocumentRow, type EntityRow, type TraversalPath,
} from './store.js';
import { traverseNeo4j } from './neo4j.js';

export interface RetrievedChunk {
  chunkId: number;
  documentId: number;
  documentTitle: string;
  source: string;
  ordinal: number;
  text: string;
  /** Normalised fused score, 1.0 = best hit. */
  score: number;
  vectorRank: number | null;
  keywordRank: number | null;
  via: ('vector' | 'keyword' | 'graph')[];
}

export interface GraphNode { id: number; name: string; class: string; depth: number }
export interface GraphContext {
  engine: 'neo4j' | 'sqlite' | 'none';
  seeds: GraphNode[];
  nodes: GraphNode[];
  edges: TraversalPath[];
}

export interface RetrievalResult {
  question: string;
  redactedQuestion: string;
  redactions: number;
  chunks: RetrievedChunk[];
  graph: GraphContext;
  embedder: { kind: string; model: string };
  fts: boolean;
  timings: { embedMs: number; vectorMs: number; keywordMs: number; graphMs: number };
}

export interface RetrieveOptions {
  maxChunks?: number;
  minScore?: number;
  hybrid?: boolean;
  graph?: boolean;
}

const RRF_K = 60;
const STOPWORDS = new Set(['the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'is', 'are', 'was', 'were', 'be', 'it', 'this', 'that', 'with', 'as', 'by', 'at', 'from', 'do', 'does', 'how', 'what', 'which', 'who', 'when', 'where', 'why', 'can', 'i', 'my', 'you', 'your', 'we', 'our']);

export function questionTerms(question: string, max = 12): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of question.toLowerCase().matchAll(/[\p{L}\p{N}_][\p{L}\p{N}_.-]*/gu)) {
    const t = m[0].replace(/^[.-]+|[.-]+$/g, '');
    if (t.length < 2 || STOPWORDS.has(t) || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
    if (out.length >= max) break;
  }
  return out;
}

/** FTS5 MATCH expression: quoted terms OR-ed for recall; ranking does the rest. */
export function ftsQuery(question: string): string | null {
  const terms = questionTerms(question);
  if (terms.length === 0) return null;
  return terms.map(t => `"${t.replace(/"/g, '""')}"`).join(' OR ');
}

interface RawHit { chunkId: number; documentId: number }

function keywordSearch(baseId: number, question: string, limit: number): RawHit[] {
  const db = getDb();
  if (hasFts()) {
    const q = ftsQuery(question);
    if (!q) return [];
    try {
      return (db.prepare(`
        SELECT f.rowid AS chunkId, c.document_id AS documentId
        FROM knowledge_chunks_fts f JOIN knowledge_chunks c ON c.id = f.rowid
        WHERE knowledge_chunks_fts MATCH ? AND c.base_id = ?
        ORDER BY bm25(knowledge_chunks_fts) LIMIT ?
      `).all(q, baseId, limit) as RawHit[]);
    } catch (err: any) {
      console.warn(`[knowledge] FTS query failed, falling back to LIKE: ${String(err?.message ?? err).slice(0, 120)}`);
    }
  }
  // No FTS5 (or a MATCH the tokenizer rejected): count term occurrences.
  const terms = questionTerms(question);
  if (terms.length === 0) return [];
  const rows = db.prepare('SELECT id, document_id, lower(text) AS t FROM knowledge_chunks WHERE base_id = ? LIMIT 5000').all(baseId) as { id: number; document_id: number; t: string }[];
  const scored = rows
    .map(r => ({ chunkId: r.id, documentId: r.document_id, score: terms.reduce((n, term) => n + (r.t.includes(term) ? 1 : 0), 0) }))
    .filter(r => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
  return scored.map(({ chunkId, documentId }) => ({ chunkId, documentId }));
}

interface Fused { chunkId: number; documentId: number; score: number; vectorRank: number | null; keywordRank: number | null; via: Set<'vector' | 'keyword' | 'graph'> }

function fuse(vector: RawHit[], keyword: RawHit[]): Map<number, Fused> {
  const out = new Map<number, Fused>();
  const add = (hits: RawHit[], kind: 'vector' | 'keyword') => {
    hits.forEach((h, i) => {
      const rank = i + 1;
      let f = out.get(h.chunkId);
      if (!f) { f = { chunkId: h.chunkId, documentId: h.documentId, score: 0, vectorRank: null, keywordRank: null, via: new Set() }; out.set(h.chunkId, f); }
      f.score += 1 / (RRF_K + rank);
      f.via.add(kind);
      if (kind === 'vector') f.vectorRank = rank; else f.keywordRank = rank;
    });
  };
  add(vector, 'vector');
  add(keyword, 'keyword');
  return out;
}

/** Entities whose canonical name appears in the question, longest first. */
export function linkEntities(baseId: number, question: string, limit: number): EntityRow[] {
  const q = question.toLowerCase().replace(/\s+/g, ' ');
  return getDb().prepare(`
    SELECT * FROM knowledge_entities
    WHERE base_id = ? AND length(canonical) >= 3 AND instr(?, canonical) > 0
    ORDER BY length(canonical) DESC, mention_count DESC LIMIT ?
  `).all(baseId, q, limit) as EntityRow[];
}

function seedsFromChunks(chunkIds: number[], limit: number): number[] {
  if (chunkIds.length === 0 || limit <= 0) return [];
  const ph = chunkIds.map(() => '?').join(',');
  const rows = getDb().prepare(`
    SELECT m.entity_id AS id, SUM(m.confidence) AS w FROM knowledge_mentions m WHERE m.chunk_id IN (${ph})
    GROUP BY m.entity_id ORDER BY w DESC LIMIT ?
  `).all(...chunkIds, limit) as { id: number }[];
  return rows.map(r => r.id);
}

export async function retrieve(base: BaseRow, question: string, opts: RetrieveOptions = {}): Promise<RetrievalResult> {
  const policy = getPolicy();
  const maxChunks = Math.max(1, Math.min(opts.maxChunks ?? policy.retrieval.max_chunks, 50));
  const minScore = opts.minScore ?? policy.retrieval.min_score;
  const hybrid = opts.hybrid ?? policy.retrieval.hybrid;
  const graphEnabled = opts.graph ?? policy.retrieval.graph.enabled;
  const candidateN = Math.max(20, maxChunks * 3);
  const timings = { embedMs: 0, vectorMs: 0, keywordMs: 0, graphMs: 0 };

  const red = redactText(question, policy);
  const q = red.text;

  const embedder = embedderFor(base.embedder || 'hash', base.embedding_model);
  let t = Date.now();
  const [qv] = await embedder.embed([q]);
  timings.embedMs = Date.now() - t;

  t = Date.now();
  const vectorHits = vectorSearch(base.id, normalize(qv), candidateN);
  timings.vectorMs = Date.now() - t;

  t = Date.now();
  const keywordHits = hybrid ? keywordSearch(base.id, q, candidateN) : [];
  timings.keywordMs = Date.now() - t;

  const fused = fuse(vectorHits, keywordHits);
  const ranked = () => [...fused.values()].sort((a, b) => b.score - a.score);

  const graph: GraphContext = { engine: 'none', seeds: [], nodes: [], edges: [] };
  if (graphEnabled) {
    t = Date.now();
    const maxEntities = policy.retrieval.graph.max_entities;
    const linked = linkEntities(base.id, q, maxEntities);
    const seedIds = new Set(linked.map(e => e.id));
    for (const id of seedsFromChunks(ranked().slice(0, 3).map(f => f.chunkId), Math.max(0, maxEntities - seedIds.size))) seedIds.add(id);
    if (seedIds.size > 0) {
      const seeds = [...seedIds];
      let walk: { nodes: number[]; edges: TraversalPath[] } | null = null;
      try {
        walk = await traverseNeo4j(base.id, seeds, policy.retrieval.graph.max_hops, 50);
        if (walk) graph.engine = 'neo4j';
      } catch (err: any) {
        console.warn(`[knowledge/neo4j] traversal failed, using SQLite: ${String(err?.message ?? err).slice(0, 120)}`);
      }
      if (!walk) {
        walk = traverse(base.id, seeds, policy.retrieval.graph.max_hops, 50);
        graph.engine = 'sqlite';
      }
      const depthOf = new Map<number, number>();
      for (const e of walk.edges) {
        depthOf.set(e.fromId, Math.min(depthOf.get(e.fromId) ?? e.depth, e.depth));
        depthOf.set(e.toId, Math.min(depthOf.get(e.toId) ?? e.depth, e.depth));
      }
      const node = (id: number, depth: number): GraphNode | null => {
        const row = getEntity(id);
        return row ? { id, name: row.name, class: row.class, depth } : null;
      };
      graph.seeds = seeds.map(id => node(id, 0)).filter((n): n is GraphNode => !!n);
      graph.nodes = walk.nodes.map(id => node(id, seedIds.has(id) ? 0 : (depthOf.get(id) ?? 1))).filter((n): n is GraphNode => !!n);
      graph.edges = walk.edges;

      // Evidence: chunks that back the traversed edges first, then chunks
      // mentioning the neighbourhood. Present hits get a support bonus; new
      // ones enter just below the weakest direct hit.
      const budget = policy.retrieval.graph.max_context_chunks;
      const scores = [...fused.values()].map(f => f.score);
      const floor = scores.length ? Math.min(...scores) : 1 / (RRF_K + candidateN);
      const evidence: number[] = [];
      for (const e of walk.edges) if (e.chunkId != null && !evidence.includes(e.chunkId)) evidence.push(e.chunkId);
      for (const m of chunksMentioning(walk.nodes, budget * 3)) if (!evidence.includes(m.chunk_id)) evidence.push(m.chunk_id);
      let added = 0;
      for (const chunkId of evidence) {
        const existing = fused.get(chunkId);
        if (existing) {
          existing.score += 1 / (RRF_K + 10);
          existing.via.add('graph');
          continue;
        }
        if (added >= budget) continue;
        const row = getChunks([chunkId])[0];
        if (!row || row.base_id !== base.id) continue;
        fused.set(chunkId, { chunkId, documentId: row.document_id, score: floor * 0.9 - added * 1e-6, vectorRank: null, keywordRank: null, via: new Set(['graph']) });
        added++;
      }
    }
    timings.graphMs = Date.now() - t;
  }

  const ordered = ranked();
  const top = ordered[0]?.score ?? 0;
  const selected = ordered
    .map(f => ({ ...f, score: top > 0 ? f.score / top : 0 }))
    .filter(f => f.score >= minScore)
    .slice(0, maxChunks);

  const chunkRows = new Map(getChunks(selected.map(s => s.chunkId)).map(c => [c.id, c]));
  const docs = new Map<number, DocumentRow | undefined>();
  const doc = (id: number) => {
    if (!docs.has(id)) docs.set(id, getDocument(id));
    return docs.get(id);
  };
  const chunks: RetrievedChunk[] = [];
  for (const s of selected) {
    const c = chunkRows.get(s.chunkId);
    if (!c) continue;
    const d = doc(c.document_id);
    chunks.push({
      chunkId: c.id, documentId: c.document_id, documentTitle: d?.title ?? '', source: d?.source ?? '', ordinal: c.ordinal, text: c.text,
      score: Number(s.score.toFixed(4)), vectorRank: s.vectorRank, keywordRank: s.keywordRank, via: [...s.via],
    });
  }

  return {
    question, redactedQuestion: q, redactions: red.total, chunks, graph,
    embedder: { kind: embedder.kind, model: embedder.model }, fts: hasFts(), timings,
  };
}
