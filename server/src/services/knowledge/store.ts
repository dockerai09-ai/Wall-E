// Data access for the knowledge tables. Typed row interfaces plus prepared
// statements against the Db abstraction — the same shape services/embeddings.ts
// uses. No ORM. Everything that touches chunk embeddings invalidates the
// vector cache (vector.ts).

import crypto from 'crypto';
import { getDb } from '../../db/index.js';
import { KnowledgeError } from './config.js';
import { invalidateVectorCache, encodeVector, normalize } from './vector.js';
import type { TextChunk } from './chunker.js';

export interface BaseRow {
  id: number;
  slug: string;
  name: string;
  description: string;
  profile_id: number | null;
  shared: number;
  embedder: string;
  embedding_model: string;
  embedding_dims: number;
  created_at_ms: number;
  updated_at_ms: number;
}

export interface DocumentRow {
  id: number;
  base_id: number;
  title: string;
  source: string;
  content_type: string;
  content_hash: string;
  byte_size: number;
  chunk_count: number;
  status: 'indexing' | 'ready' | 'error' | 'deleted';
  error: string | null;
  metadata_json: string;
  redactions: number;
  created_at_ms: number;
  updated_at_ms: number;
  deleted_at_ms: number | null;
}

export interface ChunkRow {
  id: number;
  base_id: number;
  document_id: number;
  ordinal: number;
  text: string;
  token_count: number;
  char_start: number;
  char_end: number;
  content_hash: string;
  embedding: Buffer | null;
  embedding_model: string | null;
  embedding_dims: number | null;
  graph_status: 'pending' | 'done' | 'skipped' | 'error';
  created_at_ms: number;
}

export interface EntityRow {
  id: number;
  base_id: number;
  class: string;
  name: string;
  canonical: string;
  properties_json: string;
  mention_count: number;
  created_at_ms: number;
}

export interface RelationRow {
  id: number;
  base_id: number;
  from_entity_id: number;
  to_entity_id: number;
  type: string;
  confidence: number;
  chunk_id: number | null;
  properties_json: string;
  created_at_ms: number;
}

export interface QueryRow {
  id: string;
  base_id: number;
  kind: string;
  actor: string;
  question: string;
  question_hash: string;
  retrieval_json: string;
  graph_json: string;
  prompt_hash: string | null;
  platform: string | null;
  model_id: string | null;
  answer: string | null;
  answer_hash: string | null;
  citations_json: string;
  usage_json: string;
  governance_json: string;
  status: 'ok' | 'refused' | 'error';
  error: string | null;
  latency_ms: number;
  created_at_ms: number;
}

export interface EvalRunRow {
  id: string;
  base_id: number;
  dataset: string;
  status: 'running' | 'done' | 'error';
  cases: number;
  metrics_json: string;
  results_json: string;
  config_json: string;
  error: string | null;
  started_at_ms: number;
  finished_at_ms: number | null;
}

export function sha256(text: string): string {
  return crypto.createHash('sha256').update(text).digest('hex');
}

export function canonicalName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

const SLUG_RE = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;

export function slugify(input: string): string {
  const slug = input.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64);
  if (!SLUG_RE.test(slug)) throw new KnowledgeError(`'${input}' cannot be turned into a valid slug`);
  return slug;
}

// ----------------------------------------------------------------- bases ----

export interface CreateBaseInput {
  slug: string;
  name: string;
  description?: string;
  profileId?: number | null;
  shared?: boolean;
  embedder: string;
  embeddingModel: string;
}

export function createBase(input: CreateBaseInput): BaseRow {
  const slug = slugify(input.slug);
  const now = Date.now();
  try {
    const r = getDb().prepare(`
      INSERT INTO knowledge_bases (slug, name, description, profile_id, shared, embedder, embedding_model, embedding_dims, created_at_ms, updated_at_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
    `).run(slug, input.name.trim() || slug, input.description ?? '', input.profileId ?? null, input.shared === false ? 0 : 1, input.embedder, input.embeddingModel, now, now);
    return getBase(Number(r.lastInsertRowid))!;
  } catch (err: any) {
    if (/UNIQUE/i.test(String(err?.message))) throw new KnowledgeError(`a knowledge base with slug '${slug}' already exists`, 409, 'conflict');
    throw err;
  }
}

export function getBase(id: number): BaseRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_bases WHERE id = ?').get(id) as BaseRow | undefined;
}

export function getBaseBySlug(slug: string): BaseRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_bases WHERE slug = ?').get(slug) as BaseRow | undefined;
}

/** Accepts a numeric id or a slug. */
export function resolveBase(ref: string | number): BaseRow | undefined {
  if (typeof ref === 'number') return getBase(ref);
  if (/^\d+$/.test(ref)) return getBase(Number(ref));
  return getBaseBySlug(ref);
}

export function listBases(): BaseRow[] {
  return getDb().prepare('SELECT * FROM knowledge_bases ORDER BY created_at_ms ASC').all() as BaseRow[];
}

export function touchBase(id: number): void {
  getDb().prepare('UPDATE knowledge_bases SET updated_at_ms = ? WHERE id = ?').run(Date.now(), id);
}

export function setBaseEmbedding(id: number, embedder: string, model: string, dims: number): void {
  getDb().prepare('UPDATE knowledge_bases SET embedder = ?, embedding_model = ?, embedding_dims = ?, updated_at_ms = ? WHERE id = ?')
    .run(embedder, model, dims, Date.now(), id);
}

export function deleteBase(id: number): boolean {
  const changes = getDb().prepare('DELETE FROM knowledge_bases WHERE id = ?').run(id).changes;
  invalidateVectorCache(id);
  return changes > 0;
}

export interface BaseStats {
  documents: number;
  documentsReady: number;
  chunks: number;
  chunksEmbedded: number;
  chunksGraphPending: number;
  entities: number;
  relations: number;
  queries: number;
}

export function baseStats(id: number): BaseStats {
  const db = getDb();
  const one = (sql: string) => Number((db.prepare(sql).get(id) as { n: number }).n);
  return {
    documents: one("SELECT COUNT(*) n FROM knowledge_documents WHERE base_id = ? AND status != 'deleted'"),
    documentsReady: one("SELECT COUNT(*) n FROM knowledge_documents WHERE base_id = ? AND status = 'ready'"),
    chunks: one('SELECT COUNT(*) n FROM knowledge_chunks WHERE base_id = ?'),
    chunksEmbedded: one('SELECT COUNT(*) n FROM knowledge_chunks WHERE base_id = ? AND embedding IS NOT NULL'),
    chunksGraphPending: one("SELECT COUNT(*) n FROM knowledge_chunks WHERE base_id = ? AND graph_status = 'pending'"),
    entities: one('SELECT COUNT(*) n FROM knowledge_entities WHERE base_id = ?'),
    relations: one('SELECT COUNT(*) n FROM knowledge_relations WHERE base_id = ?'),
    queries: one('SELECT COUNT(*) n FROM knowledge_queries WHERE base_id = ?'),
  };
}

// ------------------------------------------------------------- documents ----

export interface CreateDocumentInput {
  baseId: number;
  title: string;
  source?: string;
  contentType?: string;
  contentHash: string;
  byteSize: number;
  metadata?: Record<string, unknown>;
  redactions?: number;
  chunks: TextChunk[];
}

/** Inserts the document and all of its chunks in one transaction. */
export function createDocumentWithChunks(input: CreateDocumentInput): DocumentRow {
  const db = getDb();
  const now = Date.now();
  const insertDoc = db.prepare(`
    INSERT INTO knowledge_documents (base_id, title, source, content_type, content_hash, byte_size, chunk_count, status, metadata_json, redactions, created_at_ms, updated_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, 'indexing', ?, ?, ?, ?)
  `);
  const insertChunk = db.prepare(`
    INSERT INTO knowledge_chunks (base_id, document_id, ordinal, text, token_count, char_start, char_end, content_hash, created_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const run = db.transaction(() => {
    const r = insertDoc.run(
      input.baseId, input.title, input.source ?? '', input.contentType ?? 'text/plain', input.contentHash, input.byteSize,
      input.chunks.length, JSON.stringify(input.metadata ?? {}), input.redactions ?? 0, now, now,
    );
    const docId = Number(r.lastInsertRowid);
    for (const c of input.chunks) {
      insertChunk.run(input.baseId, docId, c.ordinal, c.text, c.tokenCount, c.charStart, c.charEnd, sha256(c.text), now);
    }
    return docId;
  });
  try {
    const docId = run();
    touchBase(input.baseId);
    invalidateVectorCache(input.baseId);
    return getDocument(docId)!;
  } catch (err: any) {
    if (/UNIQUE/i.test(String(err?.message))) throw new KnowledgeError('an identical document already exists in this base', 409, 'conflict');
    throw err;
  }
}

export function getDocument(id: number): DocumentRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_documents WHERE id = ?').get(id) as DocumentRow | undefined;
}

export function findDocumentByHash(baseId: number, contentHash: string): DocumentRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_documents WHERE base_id = ? AND content_hash = ?').get(baseId, contentHash) as DocumentRow | undefined;
}

export function listDocuments(baseId: number, opts: { includeDeleted?: boolean; limit?: number } = {}): DocumentRow[] {
  const where = opts.includeDeleted ? '' : "AND status != 'deleted'";
  return getDb().prepare(`SELECT * FROM knowledge_documents WHERE base_id = ? ${where} ORDER BY created_at_ms DESC LIMIT ?`)
    .all(baseId, Math.min(Math.max(opts.limit ?? 200, 1), 2000)) as DocumentRow[];
}

export function setDocumentStatus(id: number, status: DocumentRow['status'], error: string | null = null): void {
  getDb().prepare('UPDATE knowledge_documents SET status = ?, error = ?, updated_at_ms = ? WHERE id = ?').run(status, error, Date.now(), id);
}

/**
 * Right-to-be-forgotten delete: chunks, embeddings, mentions and evidenced
 * relations go (cascade); the document row stays as a tombstone so provenance
 * records that cite it can say "deleted" instead of dangling.
 */
export function deleteDocument(id: number): DocumentRow | undefined {
  const db = getDb();
  const doc = getDocument(id);
  if (!doc) return undefined;
  db.transaction(() => {
    db.prepare('DELETE FROM knowledge_chunks WHERE document_id = ?').run(id);
    // Entities nothing mentions any more are orphans; drop them so the graph
    // does not keep names the source no longer supports.
    db.prepare(`
      DELETE FROM knowledge_entities WHERE base_id = ? AND id NOT IN (SELECT entity_id FROM knowledge_mentions)
        AND id NOT IN (SELECT from_entity_id FROM knowledge_relations) AND id NOT IN (SELECT to_entity_id FROM knowledge_relations)
    `).run(doc.base_id);
    db.prepare("UPDATE knowledge_documents SET status = 'deleted', chunk_count = 0, deleted_at_ms = ?, updated_at_ms = ? WHERE id = ?")
      .run(Date.now(), Date.now(), id);
  })();
  touchBase(doc.base_id);
  invalidateVectorCache(doc.base_id);
  return getDocument(id);
}

// ---------------------------------------------------------------- chunks ----

export function getChunk(id: number): ChunkRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_chunks WHERE id = ?').get(id) as ChunkRow | undefined;
}

export function getChunks(ids: number[]): ChunkRow[] {
  if (ids.length === 0) return [];
  const placeholders = ids.map(() => '?').join(',');
  return getDb().prepare(`SELECT * FROM knowledge_chunks WHERE id IN (${placeholders})`).all(...ids) as ChunkRow[];
}

export function listDocumentChunks(documentId: number): ChunkRow[] {
  return getDb().prepare('SELECT * FROM knowledge_chunks WHERE document_id = ? ORDER BY ordinal').all(documentId) as ChunkRow[];
}

export function pendingEmbeddingChunks(baseId: number, limit: number): ChunkRow[] {
  return getDb().prepare('SELECT * FROM knowledge_chunks WHERE base_id = ? AND embedding IS NULL ORDER BY id LIMIT ?').all(baseId, limit) as ChunkRow[];
}

export function pendingGraphChunks(baseId: number, limit: number): ChunkRow[] {
  return getDb().prepare(
    "SELECT * FROM knowledge_chunks WHERE base_id = ? AND graph_status = 'pending' AND embedding IS NOT NULL ORDER BY id LIMIT ?",
  ).all(baseId, limit) as ChunkRow[];
}

export function basesWithPendingWork(): number[] {
  const rows = getDb().prepare(`
    SELECT DISTINCT base_id FROM knowledge_chunks WHERE embedding IS NULL OR graph_status = 'pending'
    UNION SELECT DISTINCT base_id FROM knowledge_documents WHERE status = 'indexing'
  `).all() as { base_id: number }[];
  return rows.map(r => r.base_id);
}

export function storeEmbeddings(baseId: number, rows: { chunkId: number; vector: number[] }[], model: string): void {
  const db = getDb();
  const stmt = db.prepare('UPDATE knowledge_chunks SET embedding = ?, embedding_model = ?, embedding_dims = ? WHERE id = ?');
  db.transaction(() => {
    for (const r of rows) {
      const vec = normalize(r.vector);
      stmt.run(encodeVector(vec), model, vec.length, r.chunkId);
    }
  })();
  invalidateVectorCache(baseId);
}

export function clearEmbeddings(baseId: number): number {
  const changes = getDb().prepare("UPDATE knowledge_chunks SET embedding = NULL, embedding_model = NULL, embedding_dims = NULL, graph_status = 'pending' WHERE base_id = ?").run(baseId).changes;
  invalidateVectorCache(baseId);
  return changes;
}

/** Put failed/skipped chunks back in the extraction queue without re-embedding. */
export function resetGraphState(baseId: number): number {
  const changes = getDb().prepare("UPDATE knowledge_chunks SET graph_status = 'pending' WHERE base_id = ? AND graph_status IN ('error', 'skipped')").run(baseId).changes;
  if (changes > 0) getDb().prepare("UPDATE knowledge_documents SET status = 'indexing', updated_at_ms = ? WHERE base_id = ? AND status = 'ready'").run(Date.now(), baseId);
  return changes;
}

export function setChunkGraphStatus(chunkId: number, status: ChunkRow['graph_status']): void {
  getDb().prepare('UPDATE knowledge_chunks SET graph_status = ? WHERE id = ?').run(status, chunkId);
}

/** Documents whose chunks are all embedded (and graph-processed when required). */
export function finalizeReadyDocuments(baseId: number, requireGraph: boolean): number {
  const graphClause = requireGraph ? "OR c.graph_status = 'pending'" : '';
  return getDb().prepare(`
    UPDATE knowledge_documents SET status = 'ready', error = NULL, updated_at_ms = ?
    WHERE base_id = ? AND status = 'indexing' AND NOT EXISTS (
      SELECT 1 FROM knowledge_chunks c WHERE c.document_id = knowledge_documents.id AND (c.embedding IS NULL ${graphClause})
    )
  `).run(Date.now(), baseId).changes;
}

// ----------------------------------------------------------------- graph ----

export function upsertEntity(baseId: number, cls: string, name: string, properties: Record<string, unknown> = {}): EntityRow {
  const db = getDb();
  const canonical = canonicalName(name);
  const existing = db.prepare('SELECT * FROM knowledge_entities WHERE base_id = ? AND class = ? AND canonical = ?').get(baseId, cls, canonical) as EntityRow | undefined;
  if (existing) {
    if (Object.keys(properties).length > 0) {
      const merged = { ...JSON.parse(existing.properties_json || '{}'), ...properties };
      db.prepare('UPDATE knowledge_entities SET properties_json = ? WHERE id = ?').run(JSON.stringify(merged), existing.id);
      existing.properties_json = JSON.stringify(merged);
    }
    return existing;
  }
  const r = db.prepare(`
    INSERT INTO knowledge_entities (base_id, class, name, canonical, properties_json, mention_count, created_at_ms)
    VALUES (?, ?, ?, ?, ?, 0, ?)
  `).run(baseId, cls, name.trim(), canonical, JSON.stringify(properties), Date.now());
  return db.prepare('SELECT * FROM knowledge_entities WHERE id = ?').get(Number(r.lastInsertRowid)) as EntityRow;
}

export function addMention(chunkId: number, entityId: number, confidence: number): void {
  const db = getDb();
  const r = db.prepare('INSERT OR IGNORE INTO knowledge_mentions (chunk_id, entity_id, confidence) VALUES (?, ?, ?)').run(chunkId, entityId, confidence);
  if (r.changes > 0) db.prepare('UPDATE knowledge_entities SET mention_count = mention_count + 1 WHERE id = ?').run(entityId);
}

export function addRelation(baseId: number, fromId: number, toId: number, type: string, confidence: number, chunkId: number | null, properties: Record<string, unknown> = {}): boolean {
  const r = getDb().prepare(`
    INSERT OR IGNORE INTO knowledge_relations (base_id, from_entity_id, to_entity_id, type, confidence, chunk_id, properties_json, created_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(baseId, fromId, toId, type, confidence, chunkId, JSON.stringify(properties), Date.now());
  return r.changes > 0;
}

export function getEntity(id: number): EntityRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_entities WHERE id = ?').get(id) as EntityRow | undefined;
}

export function listEntities(baseId: number, opts: { q?: string; cls?: string; limit?: number } = {}): EntityRow[] {
  const clauses = ['base_id = ?'];
  const params: unknown[] = [baseId];
  if (opts.q) { clauses.push('canonical LIKE ?'); params.push(`%${canonicalName(opts.q)}%`); }
  if (opts.cls) { clauses.push('class = ?'); params.push(opts.cls); }
  params.push(Math.min(Math.max(opts.limit ?? 100, 1), 5000));
  return getDb().prepare(`SELECT * FROM knowledge_entities WHERE ${clauses.join(' AND ')} ORDER BY mention_count DESC, name ASC LIMIT ?`).all(...params) as EntityRow[];
}

export interface NeighborEdge {
  relation: RelationRow;
  from: EntityRow;
  to: EntityRow;
}

export function entityEdges(entityId: number, limit = 200): NeighborEdge[] {
  const db = getDb();
  const rels = db.prepare('SELECT * FROM knowledge_relations WHERE from_entity_id = ? OR to_entity_id = ? ORDER BY confidence DESC LIMIT ?').all(entityId, entityId, limit) as RelationRow[];
  const cache = new Map<number, EntityRow>();
  const ent = (id: number) => {
    let e = cache.get(id);
    if (!e) { e = getEntity(id)!; cache.set(id, e); }
    return e;
  };
  return rels.map(relation => ({ relation, from: ent(relation.from_entity_id), to: ent(relation.to_entity_id) }));
}

export interface TraversalPath { fromId: number; toId: number; type: string; chunkId: number | null; depth: number; confidence: number }

/** k-hop neighbourhood over the SQLite relation table (recursive CTE). */
export function traverse(baseId: number, seedIds: number[], maxHops: number, maxNodes = 50): { nodes: number[]; edges: TraversalPath[] } {
  if (seedIds.length === 0) return { nodes: [], edges: [] };
  const placeholders = seedIds.map(() => '?').join(',');
  const rows = getDb().prepare(`
    WITH RECURSIVE walk(entity_id, depth) AS (
      SELECT id, 0 FROM knowledge_entities WHERE base_id = ? AND id IN (${placeholders})
      UNION
      SELECT CASE WHEN r.from_entity_id = w.entity_id THEN r.to_entity_id ELSE r.from_entity_id END, w.depth + 1
      FROM walk w JOIN knowledge_relations r ON (r.from_entity_id = w.entity_id OR r.to_entity_id = w.entity_id)
      WHERE w.depth < ? AND r.base_id = ?
    )
    SELECT entity_id, MIN(depth) depth FROM walk GROUP BY entity_id ORDER BY depth LIMIT ?
  `).all(baseId, ...seedIds, maxHops, baseId, maxNodes) as { entity_id: number; depth: number }[];
  const nodes = rows.map(r => r.entity_id);
  if (nodes.length === 0) return { nodes: [], edges: [] };
  const nodeSet = new Set(nodes);
  const depthOf = new Map(rows.map(r => [r.entity_id, r.depth]));
  const ph = nodes.map(() => '?').join(',');
  const rels = getDb().prepare(
    `SELECT * FROM knowledge_relations WHERE base_id = ? AND from_entity_id IN (${ph}) AND to_entity_id IN (${ph}) ORDER BY confidence DESC LIMIT 500`,
  ).all(baseId, ...nodes, ...nodes) as RelationRow[];
  const edges = rels
    .filter(r => nodeSet.has(r.from_entity_id) && nodeSet.has(r.to_entity_id))
    .map(r => ({
      fromId: r.from_entity_id, toId: r.to_entity_id, type: r.type, chunkId: r.chunk_id, confidence: r.confidence,
      depth: Math.max(depthOf.get(r.from_entity_id) ?? 0, depthOf.get(r.to_entity_id) ?? 0),
    }));
  return { nodes, edges };
}

export function chunksMentioning(entityIds: number[], limit: number): { chunk_id: number; entity_id: number }[] {
  if (entityIds.length === 0 || limit <= 0) return [];
  const ph = entityIds.map(() => '?').join(',');
  return getDb().prepare(`SELECT chunk_id, entity_id FROM knowledge_mentions WHERE entity_id IN (${ph}) ORDER BY confidence DESC LIMIT ?`)
    .all(...entityIds, limit) as { chunk_id: number; entity_id: number }[];
}

export interface GraphStats { entities: number; relations: number; byClass: Record<string, number>; byType: Record<string, number>; catchAllRatio: number }

export function graphStats(baseId: number, catchAllType = 'RELATED_TO'): GraphStats {
  const db = getDb();
  const byClass: Record<string, number> = {};
  for (const r of db.prepare('SELECT class, COUNT(*) n FROM knowledge_entities WHERE base_id = ? GROUP BY class').all(baseId) as { class: string; n: number }[]) byClass[r.class] = r.n;
  const byType: Record<string, number> = {};
  let total = 0;
  for (const r of db.prepare('SELECT type, COUNT(*) n FROM knowledge_relations WHERE base_id = ? GROUP BY type').all(baseId) as { type: string; n: number }[]) { byType[r.type] = r.n; total += r.n; }
  return {
    entities: Object.values(byClass).reduce((a, b) => a + b, 0),
    relations: total,
    byClass,
    byType,
    catchAllRatio: total === 0 ? 0 : (byType[catchAllType] ?? 0) / total,
  };
}

// --------------------------------------------------------------- queries ----

export function insertQuery(row: Omit<QueryRow, 'created_at_ms'> & { created_at_ms?: number }): void {
  getDb().prepare(`
    INSERT INTO knowledge_queries (id, base_id, kind, actor, question, question_hash, retrieval_json, graph_json, prompt_hash, platform, model_id, answer, answer_hash, citations_json, usage_json, governance_json, status, error, latency_ms, created_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    row.id, row.base_id, row.kind, row.actor, row.question, row.question_hash, row.retrieval_json, row.graph_json, row.prompt_hash,
    row.platform, row.model_id, row.answer, row.answer_hash, row.citations_json, row.usage_json, row.governance_json, row.status,
    row.error, row.latency_ms, row.created_at_ms ?? Date.now(),
  );
}

export function getQuery(id: string): QueryRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_queries WHERE id = ?').get(id) as QueryRow | undefined;
}

export function listQueries(baseId: number, limit = 50): QueryRow[] {
  return getDb().prepare('SELECT * FROM knowledge_queries WHERE base_id = ? ORDER BY created_at_ms DESC LIMIT ?').all(baseId, Math.min(Math.max(limit, 1), 500)) as QueryRow[];
}

// ------------------------------------------------------------- eval runs ----

export function insertEvalRun(row: Pick<EvalRunRow, 'id' | 'base_id' | 'dataset' | 'cases' | 'config_json'>): void {
  getDb().prepare(`
    INSERT INTO knowledge_eval_runs (id, base_id, dataset, status, cases, config_json, started_at_ms)
    VALUES (?, ?, ?, 'running', ?, ?, ?)
  `).run(row.id, row.base_id, row.dataset, row.cases, row.config_json, Date.now());
}

export function finishEvalRun(id: string, status: EvalRunRow['status'], metrics: unknown, results: unknown, error: string | null = null): void {
  getDb().prepare('UPDATE knowledge_eval_runs SET status = ?, metrics_json = ?, results_json = ?, error = ?, finished_at_ms = ? WHERE id = ?')
    .run(status, JSON.stringify(metrics), JSON.stringify(results), error, Date.now(), id);
}

export function getEvalRun(id: string): EvalRunRow | undefined {
  return getDb().prepare('SELECT * FROM knowledge_eval_runs WHERE id = ?').get(id) as EvalRunRow | undefined;
}

export function listEvalRuns(baseId: number | null, limit = 50): EvalRunRow[] {
  const lim = Math.min(Math.max(limit, 1), 500);
  if (baseId == null) {
    return getDb().prepare('SELECT id, base_id, dataset, status, cases, metrics_json, config_json, error, started_at_ms, finished_at_ms, \'[]\' AS results_json FROM knowledge_eval_runs ORDER BY started_at_ms DESC LIMIT ?').all(lim) as EvalRunRow[];
  }
  return getDb().prepare('SELECT id, base_id, dataset, status, cases, metrics_json, config_json, error, started_at_ms, finished_at_ms, \'[]\' AS results_json FROM knowledge_eval_runs WHERE base_id = ? ORDER BY started_at_ms DESC LIMIT ?').all(baseId, lim) as EvalRunRow[];
}

export function hasFts(): boolean {
  const row = getDb().prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_chunks_fts'").get() as { name: string } | undefined;
  return !!row;
}
