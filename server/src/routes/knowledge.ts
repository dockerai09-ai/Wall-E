// Dashboard API for the knowledge module (/api/knowledge, session-gated in
// app.ts). Bodies are zod-validated; errors follow the OpenAI-shaped
// { error: { message, type } } convention the rest of /api uses.

import { Router } from 'express';
import type { Request, Response, NextFunction } from 'express';
import fs from 'fs';
import path from 'path';
import { z } from 'zod';
import { getKnowledgeConfig, KnowledgeError, REPO_ROOT } from '../services/knowledge/config.js';
import { chooseEmbedder } from '../services/knowledge/embedder.js';
import { getOntology, reloadOntology, ontologySummary } from '../services/knowledge/ontology.js';
import { audit, getPolicy, listAudit, policySummary, reloadPolicy } from '../services/knowledge/governance.js';
import * as store from '../services/knowledge/store.js';
import { ingestDocument } from '../services/knowledge/ingest.js';
import { retrieve } from '../services/knowledge/retrieval.js';
import { answerQuestion } from '../services/knowledge/rag.js';
import { getProvenance, toProvJson } from '../services/knowledge/provenance.js';
import { evalCaseSchema, evalReportMarkdown, parseJsonl, runEval, type EvalCase } from '../services/knowledge/evals.js';
import { indexerStatus, kickIndexer, runIndexerOnce } from '../services/knowledge/indexer.js';
import { neo4jStatus, readCypher, removeBaseFromNeo4j, removeDocumentFromNeo4j, syncOntologyToNeo4j } from '../services/knowledge/neo4j.js';
import type { SessionUser } from '../services/auth.js';

export const knowledgeRouter = Router();

type Handler = (req: Request, res: Response) => Promise<void> | void;

function fail(res: Response, status: number, message: string, type = 'invalid_request_error'): void {
  res.status(status).json({ error: { message, type } });
}

function handle(fn: Handler) {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      await fn(req, res);
    } catch (err: any) {
      if (err instanceof KnowledgeError) return fail(res, err.status, err.message, err.type);
      if (err instanceof z.ZodError) return fail(res, 400, err.issues[0]?.message ?? 'invalid request');
      next(err);
    }
  };
}

function actorOf(req: Request): string {
  const user = (req as Request & { user?: SessionUser }).user;
  return user ? `user:${user.userId}` : 'dashboard';
}

function iso(ms: number | null | undefined): string | null {
  return ms == null ? null : new Date(ms).toISOString();
}

export function baseView(b: store.BaseRow, withStats = true) {
  return {
    id: b.id, slug: b.slug, name: b.name, description: b.description, profileId: b.profile_id, shared: b.shared === 1,
    embedder: b.embedder, embeddingModel: b.embedding_model, embeddingDims: b.embedding_dims,
    createdAt: iso(b.created_at_ms), updatedAt: iso(b.updated_at_ms),
    ...(withStats ? { stats: store.baseStats(b.id) } : {}),
  };
}

export function documentView(d: store.DocumentRow) {
  return {
    id: d.id, baseId: d.base_id, title: d.title, source: d.source, contentType: d.content_type, contentHash: d.content_hash,
    byteSize: d.byte_size, chunkCount: d.chunk_count, status: d.status, error: d.error, redactions: d.redactions,
    metadata: JSON.parse(d.metadata_json || '{}'), createdAt: iso(d.created_at_ms), updatedAt: iso(d.updated_at_ms), deletedAt: iso(d.deleted_at_ms),
  };
}

function evalRunView(r: store.EvalRunRow, withResults: boolean) {
  return {
    id: r.id, baseId: r.base_id, dataset: r.dataset, status: r.status, cases: r.cases, error: r.error,
    metrics: JSON.parse(r.metrics_json || '{}'), config: JSON.parse(r.config_json || '{}'),
    startedAt: iso(r.started_at_ms), finishedAt: iso(r.finished_at_ms),
    ...(withResults ? { results: JSON.parse(r.results_json || '[]') } : {}),
  };
}

function requireBase(ref: string): store.BaseRow {
  const base = store.resolveBase(ref);
  if (!base) throw new KnowledgeError(`knowledge base '${ref}' not found`, 404, 'not_found');
  return base;
}

knowledgeRouter.use((_req: Request, res: Response, next: NextFunction) => {
  if (!getKnowledgeConfig().enabled) return fail(res, 404, 'the knowledge module is disabled (KB_ENABLED=false)', 'not_found');
  next();
});

// ---------------------------------------------------------------- status ----

knowledgeRouter.get('/status', handle((_req, res) => {
  const cfg = getKnowledgeConfig();
  const ontology = getOntology();
  const policy = getPolicy();
  res.json({
    enabled: cfg.enabled,
    embedder: { configured: cfg.embedder, model: cfg.embeddingModel },
    ontology: { name: ontology.name, version: ontology.version, hash: ontology.hash, path: ontology.path },
    governance: { version: policy.versionTag, path: policy.path },
    neo4j: neo4jStatus(),
    indexer: indexerStatus(),
    fts: store.hasFts(),
    bases: store.listBases().length,
  });
}));

// ----------------------------------------------------------------- bases ----

knowledgeRouter.get('/bases', handle((_req, res) => {
  res.json({ bases: store.listBases().map(b => baseView(b)) });
}));

const createBaseSchema = z.object({
  name: z.string().min(1).max(120),
  slug: z.string().min(1).max(64).optional(),
  description: z.string().max(2000).optional(),
  shared: z.boolean().optional(),
});

knowledgeRouter.post('/bases', handle(async (req, res) => {
  const body = createBaseSchema.parse(req.body ?? {});
  const choice = await chooseEmbedder();
  const base = store.createBase({
    slug: body.slug ?? body.name, name: body.name, description: body.description, shared: body.shared,
    embedder: choice.embedder.kind, embeddingModel: choice.embedder.model,
  });
  audit({ action: 'base.create', actor: actorOf(req), baseId: base.id, target: `base:${base.slug}`, details: { embedder: choice.embedder.model, warning: choice.warning } });
  if (choice.warning) console.warn(`[knowledge] base ${base.slug}: ${choice.warning}`);
  res.status(201).json({ base: baseView(base), warning: choice.warning });
}));

knowledgeRouter.get('/bases/:ref', handle((req, res) => {
  res.json({ base: baseView(requireBase(req.params.ref as string)) });
}));

knowledgeRouter.delete('/bases/:ref', handle(async (req, res) => {
  const policy = getPolicy();
  if (!policy.access.allow_delete) throw new KnowledgeError('deletion is disabled by the governance policy', 403, 'forbidden');
  const base = requireBase(req.params.ref as string);
  store.deleteBase(base.id);
  removeBaseFromNeo4j(base.id, base.slug).catch(() => {});
  audit({ action: 'base.delete', actor: actorOf(req), baseId: base.id, target: `base:${base.slug}` });
  res.json({ deleted: true, slug: base.slug });
}));

// ------------------------------------------------------------- documents ----

const ingestSchema = z.object({
  title: z.string().max(300).optional(),
  text: z.string().max(50 * 1024 * 1024).optional(),
  url: z.string().url().optional(),
  source: z.string().max(1000).optional(),
  contentType: z.string().max(100).optional(),
  metadata: z.record(z.unknown()).optional(),
}).refine(b => (b.text != null && b.text.trim().length > 0) || b.url != null, { message: 'text or url is required' });

knowledgeRouter.post('/bases/:ref/documents', handle(async (req, res) => {
  const base = requireBase(req.params.ref as string);
  const body = ingestSchema.parse(req.body ?? {});
  const result = await ingestDocument(base, { ...body, actor: actorOf(req) });
  res.status(result.deduplicated ? 200 : 201).json({
    document: documentView(result.document), deduplicated: result.deduplicated, redactions: result.redactions, chunks: result.chunks,
  });
}));

knowledgeRouter.get('/bases/:ref/documents', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  const includeDeleted = req.query.includeDeleted === '1' || req.query.includeDeleted === 'true';
  res.json({ documents: store.listDocuments(base.id, { includeDeleted }).map(documentView) });
}));

knowledgeRouter.get('/documents/:id', handle((req, res) => {
  const doc = store.getDocument(Number(req.params.id));
  if (!doc) throw new KnowledgeError('document not found', 404, 'not_found');
  const withChunks = req.query.chunks === '1' || req.query.chunks === 'true';
  res.json({
    document: documentView(doc),
    ...(withChunks ? {
      chunks: store.listDocumentChunks(doc.id).map(c => ({
        id: c.id, ordinal: c.ordinal, text: c.text, tokenCount: c.token_count, charStart: c.char_start, charEnd: c.char_end,
        embedded: c.embedding != null, embeddingModel: c.embedding_model, graphStatus: c.graph_status,
      })),
    } : {}),
  });
}));

knowledgeRouter.delete('/documents/:id', handle(async (req, res) => {
  const policy = getPolicy();
  if (!policy.access.allow_delete) throw new KnowledgeError('deletion is disabled by the governance policy', 403, 'forbidden');
  const doc = store.getDocument(Number(req.params.id));
  if (!doc) throw new KnowledgeError('document not found', 404, 'not_found');
  const chunkIds = store.listDocumentChunks(doc.id).map(c => c.id);
  const tombstone = store.deleteDocument(doc.id);
  removeDocumentFromNeo4j(doc.base_id, doc.id, chunkIds).catch(() => {});
  audit({ action: 'document.delete', actor: actorOf(req), baseId: doc.base_id, target: `document:${doc.id}`, details: { title: doc.title, chunks: chunkIds.length } });
  res.json({ deleted: true, document: tombstone ? documentView(tombstone) : null });
}));

const reindexSchema = z.object({ graphOnly: z.boolean().optional() });

// Full reindex clears embeddings and graph state; graphOnly re-queues only the
// chunks whose extraction failed or was skipped (no re-embedding).
knowledgeRouter.post('/bases/:ref/reindex', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  const body = reindexSchema.parse(req.body ?? {});
  const chunks = body.graphOnly ? store.resetGraphState(base.id) : store.clearEmbeddings(base.id);
  if (!body.graphOnly) store.listDocuments(base.id).forEach(d => store.setDocumentStatus(d.id, 'indexing'));
  audit({ action: body.graphOnly ? 'base.reindex.graph' : 'base.reindex', actor: actorOf(req), baseId: base.id, target: `base:${base.slug}`, details: { chunks } });
  kickIndexer();
  res.json({ reindexing: true, chunks, graphOnly: !!body.graphOnly });
}));

// Synchronous indexing pass (dashboard "index now", CLI, tests).
knowledgeRouter.post('/bases/:ref/index-now', handle(async (req, res) => {
  requireBase(req.params.ref as string);
  res.json(await runIndexerOnce());
}));

// ------------------------------------------------------- search / query ----

const searchSchema = z.object({
  question: z.string().min(1).max(4000),
  maxChunks: z.number().int().min(1).max(50).optional(),
  graph: z.boolean().optional(),
  hybrid: z.boolean().optional(),
});

knowledgeRouter.post('/bases/:ref/search', handle(async (req, res) => {
  const base = requireBase(req.params.ref as string);
  const body = searchSchema.parse(req.body ?? {});
  res.json(await retrieve(base, body.question, { maxChunks: body.maxChunks, graph: body.graph, hybrid: body.hybrid }));
}));

const querySchema = z.object({
  question: z.string().min(1).max(4000),
  maxChunks: z.number().int().min(1).max(50).optional(),
  model: z.string().max(200).nullable().optional(),
  graph: z.boolean().optional(),
  temperature: z.number().min(0).max(2).optional(),
});

knowledgeRouter.post('/bases/:ref/query', handle(async (req, res) => {
  const base = requireBase(req.params.ref as string);
  const body = querySchema.parse(req.body ?? {});
  res.json(await answerQuestion(base, body.question, { actor: actorOf(req), maxChunks: body.maxChunks, model: body.model ?? null, graph: body.graph, temperature: body.temperature }));
}));

knowledgeRouter.get('/bases/:ref/queries', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  const rows = store.listQueries(base.id, Number(req.query.limit ?? 50));
  res.json({
    queries: rows.map(r => ({
      id: r.id, kind: r.kind, actor: r.actor, question: r.question, status: r.status, error: r.error, latencyMs: r.latency_ms,
      model: r.platform && r.model_id ? { platform: r.platform, modelId: r.model_id } : null,
      citations: JSON.parse(r.citations_json || '[]').length, createdAt: iso(r.created_at_ms),
    })),
  });
}));

// ------------------------------------------------------------ provenance ----

knowledgeRouter.get('/provenance/:id', handle((req, res) => {
  const rec = getProvenance(req.params.id as string);
  if (!rec) throw new KnowledgeError('provenance record not found', 404, 'not_found');
  res.json(rec);
}));

knowledgeRouter.get('/provenance/:id/prov', handle((req, res) => {
  const rec = getProvenance(req.params.id as string);
  if (!rec) throw new KnowledgeError('provenance record not found', 404, 'not_found');
  res.json(toProvJson(rec));
}));

// ----------------------------------------------------------------- graph ----

knowledgeRouter.get('/bases/:ref/graph/stats', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  res.json(store.graphStats(base.id));
}));

knowledgeRouter.get('/bases/:ref/graph/entities', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  const q = typeof req.query.q === 'string' ? req.query.q : undefined;
  const cls = typeof req.query.class === 'string' ? req.query.class : undefined;
  const rows = store.listEntities(base.id, { q, cls, limit: Number(req.query.limit ?? 100) });
  res.json({ entities: rows.map(e => ({ id: e.id, name: e.name, class: e.class, mentions: e.mention_count, properties: JSON.parse(e.properties_json || '{}') })) });
}));

knowledgeRouter.get('/bases/:ref/graph/entities/:id', handle((req, res) => {
  const base = requireBase(req.params.ref as string);
  const entity = store.getEntity(Number(req.params.id));
  if (!entity || entity.base_id !== base.id) throw new KnowledgeError('entity not found', 404, 'not_found');
  const edges = store.entityEdges(entity.id);
  const mentions = store.chunksMentioning([entity.id], 50);
  res.json({
    entity: { id: entity.id, name: entity.name, class: entity.class, mentions: entity.mention_count, properties: JSON.parse(entity.properties_json || '{}') },
    edges: edges.map(e => ({
      id: e.relation.id, type: e.relation.type, confidence: e.relation.confidence, chunkId: e.relation.chunk_id,
      from: { id: e.from.id, name: e.from.name, class: e.from.class }, to: { id: e.to.id, name: e.to.name, class: e.to.class },
    })),
    chunkIds: mentions.map(m => m.chunk_id),
  });
}));

const cypherSchema = z.object({ cypher: z.string().min(1).max(10_000), params: z.record(z.unknown()).optional() });

knowledgeRouter.post('/graph/cypher', handle(async (req, res) => {
  const body = cypherSchema.parse(req.body ?? {});
  const rows = await readCypher(body.cypher, body.params ?? {});
  if (rows === null) throw new KnowledgeError('Neo4j is not configured or not reachable', 503, 'unavailable');
  audit({ action: 'graph.cypher', actor: actorOf(req), details: { cypher: body.cypher.slice(0, 500) } });
  res.json({ rows });
}));

// -------------------------------------------------- ontology / governance ----

knowledgeRouter.get('/ontology', handle((_req, res) => {
  res.json(ontologySummary(getOntology()));
}));

knowledgeRouter.post('/ontology/reload', handle(async (req, res) => {
  const ontology = reloadOntology();
  let synced = false;
  try { synced = await syncOntologyToNeo4j(ontology); } catch (err: any) { console.warn(`[knowledge/neo4j] ontology sync failed: ${err?.message ?? err}`); }
  audit({ action: 'ontology.reload', actor: actorOf(req), details: { hash: ontology.hash, version: ontology.version, neo4jSynced: synced } });
  res.json({ ...ontologySummary(ontology), neo4jSynced: synced });
}));

knowledgeRouter.post('/ontology/sync', handle(async (req, res) => {
  const synced = await syncOntologyToNeo4j(getOntology());
  audit({ action: 'ontology.sync', actor: actorOf(req), details: { neo4jSynced: synced } });
  res.json({ neo4jSynced: synced });
}));

knowledgeRouter.get('/governance', handle((_req, res) => {
  res.json(policySummary(getPolicy()));
}));

knowledgeRouter.post('/governance/reload', handle((req, res) => {
  const policy = reloadPolicy();
  audit({ action: 'governance.reload', actor: actorOf(req), details: { version: policy.versionTag } }, policy);
  res.json(policySummary(policy));
}));

knowledgeRouter.get('/audit', handle((req, res) => {
  const baseId = req.query.baseId != null ? Number(req.query.baseId) : undefined;
  const action = typeof req.query.action === 'string' ? req.query.action : undefined;
  const rows = listAudit({ baseId: Number.isFinite(baseId) ? baseId : undefined, action, limit: Number(req.query.limit ?? 100) });
  res.json({ audit: rows.map(r => ({ id: r.id, at: iso(r.at_ms), actor: r.actor, action: r.action, baseId: r.base_id, target: r.target, details: JSON.parse(r.details_json || '{}'), policyVersion: r.policy_version })) });
}));

// ----------------------------------------------------------------- evals ----

const EVALS_DIR = path.join(REPO_ROOT, 'knowledge', 'evals');

function loadDataset(name: string): { cases: EvalCase[]; datasetName: string } {
  const safe = name.replace(/\.jsonl$/i, '');
  if (!/^[A-Za-z0-9._-]+$/.test(safe)) throw new KnowledgeError('dataset names may only contain letters, digits, dots, dashes and underscores');
  const file = path.join(EVALS_DIR, `${safe}.jsonl`);
  if (!fs.existsSync(file)) throw new KnowledgeError(`dataset '${safe}' not found in knowledge/evals`, 404, 'not_found');
  return { cases: parseJsonl(fs.readFileSync(file, 'utf8')), datasetName: safe };
}

knowledgeRouter.get('/evals/datasets', handle((_req, res) => {
  const names = fs.existsSync(EVALS_DIR) ? fs.readdirSync(EVALS_DIR).filter(f => f.endsWith('.jsonl')).map(f => f.replace(/\.jsonl$/, '')) : [];
  res.json({ datasets: names });
}));

knowledgeRouter.get('/evals', handle((req, res) => {
  const baseRef = typeof req.query.base === 'string' ? req.query.base : null;
  const baseId = baseRef ? requireBase(baseRef).id : null;
  res.json({ runs: store.listEvalRuns(baseId, Number(req.query.limit ?? 50)).map(r => evalRunView(r, false)) });
}));

const runEvalSchema = z.object({
  base: z.string().min(1),
  dataset: z.string().min(1).max(200).optional(),
  cases: z.array(evalCaseSchema).min(1).optional(),
  judge: z.boolean().optional(),
  k: z.number().int().min(1).max(50).optional(),
  model: z.string().max(200).nullable().optional(),
}).refine(b => b.dataset || b.cases, { message: 'dataset or cases is required' });

knowledgeRouter.post('/evals/run', handle(async (req, res) => {
  const body = runEvalSchema.parse(req.body ?? {});
  const base = requireBase(body.base);
  const loaded = body.dataset
    ? loadDataset(body.dataset)
    : { cases: (body.cases ?? []).map((c, i) => ({ ...c, id: c.id ?? `case-${i + 1}` })), datasetName: 'inline' };
  const run = await runEval(base, loaded.cases, { datasetName: loaded.datasetName, actor: actorOf(req), judge: body.judge, k: body.k, model: body.model ?? null });
  res.status(201).json({ run: evalRunView(run, true) });
}));

knowledgeRouter.get('/evals/:id', handle((req, res) => {
  const run = store.getEvalRun(req.params.id as string);
  if (!run) throw new KnowledgeError('eval run not found', 404, 'not_found');
  res.json({ run: evalRunView(run, true) });
}));

knowledgeRouter.get('/evals/:id/report', handle((req, res) => {
  const run = store.getEvalRun(req.params.id as string);
  if (!run) throw new KnowledgeError('eval run not found', 404, 'not_found');
  const base = store.getBase(run.base_id);
  res.type('text/markdown').send(evalReportMarkdown(run, base?.slug ?? String(run.base_id)));
}));
