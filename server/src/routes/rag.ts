// /v1 RAG surface for API clients: unified-key or client-profile auth, same
// as the rest of /v1. A profile key's enforced system prompt is prepended to
// the answering prompt, and governance access.scope_by_profile hides bases a
// profile did not create unless they are shared.

import { Router } from 'express';
import type { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { resolveAuth, type ResolvedAuth } from '../lib/system-prompt.js';
import { extractApiToken } from './proxy.js';
import { getKnowledgeConfig, KnowledgeError } from '../services/knowledge/config.js';
import { getPolicy } from '../services/knowledge/governance.js';
import * as store from '../services/knowledge/store.js';
import { retrieve } from '../services/knowledge/retrieval.js';
import { answerQuestion, type KnowledgeAnswer } from '../services/knowledge/rag.js';
import { ingestDocument } from '../services/knowledge/ingest.js';

export const ragRouter = Router();

function fail(res: Response, status: number, message: string, type = 'invalid_request_error'): void {
  res.status(status).json({ error: { message, type } });
}

function actorOf(auth: ResolvedAuth): string {
  return auth.kind === 'unified' ? 'unified' : `profile:${auth.profileId}`;
}

export function canAccessBase(base: store.BaseRow, auth: ResolvedAuth): boolean {
  if (!getPolicy().access.scope_by_profile) return true;
  if (base.profile_id == null || base.shared === 1) return true;
  return auth.kind === 'profile' && auth.profileId === base.profile_id;
}

function visibleBase(ref: string, auth: ResolvedAuth): store.BaseRow {
  const base = store.resolveBase(ref);
  if (!base || !canAccessBase(base, auth)) throw new KnowledgeError(`knowledge base '${ref}' not found`, 404, 'not_found');
  return base;
}

type Handler = (req: Request, res: Response, auth: ResolvedAuth) => Promise<void> | void;

function handle(fn: Handler) {
  return async (req: Request, res: Response, next: NextFunction) => {
    if (!getKnowledgeConfig().enabled) return fail(res, 404, 'the knowledge module is disabled', 'not_found');
    const auth = resolveAuth(extractApiToken(req));
    if (!auth) return fail(res, 401, 'Invalid API key', 'authentication_error');
    try {
      await fn(req, res, auth);
    } catch (err: any) {
      if (err instanceof KnowledgeError) return fail(res, err.status, err.message, err.type);
      if (err instanceof z.ZodError) return fail(res, 400, err.issues[0]?.message ?? 'invalid request');
      next(err);
    }
  };
}

export function answerWire(a: KnowledgeAnswer) {
  return {
    id: a.id,
    object: 'knowledge.answer',
    knowledge_base: a.base.slug,
    question: a.question,
    answer: a.answer,
    status: a.status,
    citations: a.citations.map(c => ({ n: c.n, chunk_id: c.chunkId, document_id: c.documentId, title: c.title, source: c.source, part: c.ordinal + 1 })),
    sources: a.sources.map((s, i) => ({ n: i + 1, chunk_id: s.chunkId, document_id: s.documentId, title: s.documentTitle, source: s.source, part: s.ordinal + 1, score: s.score, via: s.via, text: s.text })),
    graph: { engine: a.graph.engine, entities: a.graph.nodes.map(n => ({ id: n.id, name: n.name, class: n.class, depth: n.depth })), edges: a.graph.edges.length },
    model: a.model ? `${a.model.platform}/${a.model.modelId}` : null,
    usage: a.usage,
    latency_ms: a.latencyMs,
    governance: a.governance,
    provenance: `/api/knowledge/provenance/${a.id}`,
  };
}

ragRouter.get('/rag/bases', handle((_req, res, auth) => {
  const bases = store.listBases().filter(b => canAccessBase(b, auth));
  res.json({ object: 'list', data: bases.map(b => ({ id: b.slug, name: b.name, description: b.description, embedder: b.embedding_model, stats: store.baseStats(b.id) })) });
}));

const querySchema = z.object({
  knowledge_base: z.string().min(1),
  question: z.string().min(1).max(4000),
  max_chunks: z.number().int().min(1).max(50).optional(),
  model: z.string().max(200).nullable().optional(),
  graph: z.boolean().optional(),
  temperature: z.number().min(0).max(2).optional(),
});

ragRouter.post('/rag/query', handle(async (req, res, auth) => {
  const body = querySchema.parse(req.body ?? {});
  const base = visibleBase(body.knowledge_base, auth);
  const answer = await answerQuestion(base, body.question, {
    actor: actorOf(auth), maxChunks: body.max_chunks, model: body.model ?? null, graph: body.graph, temperature: body.temperature, systemPrompt: auth.systemPrompt,
  });
  res.json(answerWire(answer));
}));

const searchSchema = z.object({
  knowledge_base: z.string().min(1),
  question: z.string().min(1).max(4000),
  max_chunks: z.number().int().min(1).max(50).optional(),
  graph: z.boolean().optional(),
});

ragRouter.post('/rag/search', handle(async (req, res, auth) => {
  const body = searchSchema.parse(req.body ?? {});
  const base = visibleBase(body.knowledge_base, auth);
  const r = await retrieve(base, body.question, { maxChunks: body.max_chunks, graph: body.graph });
  res.json({
    object: 'list',
    knowledge_base: base.slug,
    data: r.chunks.map((s, i) => ({ n: i + 1, chunk_id: s.chunkId, document_id: s.documentId, title: s.documentTitle, source: s.source, part: s.ordinal + 1, score: s.score, via: s.via, text: s.text })),
    graph: { engine: r.graph.engine, entities: r.graph.nodes.map(n => ({ id: n.id, name: n.name, class: n.class, depth: n.depth })), edges: r.graph.edges.length },
    embedder: r.embedder.model,
    redactions: r.redactions,
  });
}));

const ingestSchema = z.object({
  knowledge_base: z.string().min(1),
  title: z.string().max(300).optional(),
  text: z.string().min(1).max(50 * 1024 * 1024),
  source: z.string().max(1000).optional(),
  content_type: z.string().max(100).optional(),
  metadata: z.record(z.unknown()).optional(),
});

ragRouter.post('/rag/documents', handle(async (req, res, auth) => {
  const body = ingestSchema.parse(req.body ?? {});
  const base = visibleBase(body.knowledge_base, auth);
  const result = await ingestDocument(base, { title: body.title, text: body.text, source: body.source, contentType: body.content_type, metadata: body.metadata, actor: actorOf(auth) });
  res.status(result.deduplicated ? 200 : 201).json({
    object: 'knowledge.document', id: result.document.id, knowledge_base: base.slug, title: result.document.title, status: result.document.status,
    chunks: result.chunks, redactions: result.redactions, deduplicated: result.deduplicated,
  });
}));
