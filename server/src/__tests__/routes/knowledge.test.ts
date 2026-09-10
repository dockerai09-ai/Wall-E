import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Express } from 'express';
import type { ChatMessage } from '@freellmapi/shared/types.js';
import { createApp } from '../../app.js';
import { initDb, getUnifiedApiKey } from '../../db/index.js';
import { mintDashboardToken, isGatedApiPath } from '../helpers/auth.js';
import { setKnowledgeChatForTests } from '../../services/knowledge/llm.js';
import { parsePolicy, setPolicyForTests } from '../../services/knowledge/governance.js';
import { setGraphMirrorForTests } from '../../services/knowledge/neo4j.js';

let dashToken = '';

async function call(app: Express, method: string, path: string, body?: unknown, auth?: string) {
  const server = app.listen(0, '127.0.0.1');
  if (!server.listening) await new Promise<void>(resolve => server.once('listening', () => resolve()));
  const addr = server.address() as any;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (auth) headers.Authorization = `Bearer ${auth}`;
  else if (isGatedApiPath(path)) headers.Authorization = `Bearer ${dashToken}`;
  const res = await fetch(`http://127.0.0.1:${addr.port}${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await res.text();
  let data: any = null;
  try { data = JSON.parse(text); } catch { data = text; }
  server.close();
  return { status: res.status, body: data };
}

const chat = async (messages: ChatMessage[]) => {
  const system = String(messages[0].content);
  if (system.includes('information-extraction')) return { text: '{"entities":[{"name":"Wall-E","class":"Product","confidence":0.9}],"relations":[]}', platform: 'test', modelId: 'x', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
  if (system.includes('strict grader')) return { text: '{"correctness":1,"faithfulness":1,"reasoning":"ok"}', platform: 'test', modelId: 'judge', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
  return { text: `Answer from ${system.includes('ENFORCED') ? 'profile' : 'unified'} [1].`, platform: 'test', modelId: 'answerer', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
};

describe('/api/knowledge and /v1/rag', () => {
  let app: Express;

  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    process.env.KB_EMBEDDER = 'hash';
    initDb(':memory:');
    setPolicyForTests(parsePolicy('version: 1\nname: routes\n'));
    setGraphMirrorForTests(null);
    setKnowledgeChatForTests(chat);
    app = createApp();
    dashToken = mintDashboardToken();
  });

  afterEach(() => {
    setKnowledgeChatForTests(null);
    setPolicyForTests(null);
    setGraphMirrorForTests(undefined);
    delete process.env.KB_EMBEDDER;
    delete process.env.KB_ENABLED;
  });

  it('gates the dashboard surface behind a session', async () => {
    const r = await call(app, 'GET', '/api/knowledge/status', undefined, 'not-a-token');
    expect(r.status).toBe(401);
  });

  it('reports status with the shipped ontology and policy', async () => {
    const r = await call(app, 'GET', '/api/knowledge/status');
    expect(r.status).toBe(200);
    expect(r.body.enabled).toBe(true);
    expect(r.body.ontology.name).toBe('wall-e-default');
    expect(r.body.governance.version).toMatch(/^routes@1#/);
    expect(r.body.neo4j.connected).toBe(false);
    expect(r.body.fts).toBe(true);
  });

  it('runs the full lifecycle: base, ingest, index, search, query, provenance, audit, delete', async () => {
    const created = await call(app, 'POST', '/api/knowledge/bases', { name: 'Product Docs', description: 'd' });
    expect(created.status).toBe(201);
    expect(created.body.base.slug).toBe('product-docs');
    expect(created.body.base.embedder).toBe('hash');

    const dup = await call(app, 'POST', '/api/knowledge/bases', { name: 'Product Docs' });
    expect(dup.status).toBe(409);

    const bad = await call(app, 'POST', '/api/knowledge/bases/product-docs/documents', { title: 'x' });
    expect(bad.status).toBe(400);
    expect(bad.body.error.message).toMatch(/text or url/);

    const ing = await call(app, 'POST', '/api/knowledge/bases/product-docs/documents', {
      title: 'Ports', text: 'Wall-E listens on port 3001 by default. Email admin@example.com for help.', source: 'docs/ports.md',
    });
    expect(ing.status).toBe(201);
    expect(ing.body.document.status).toBe('indexing');
    expect(ing.body.redactions).toBe(1);
    expect(ing.body.chunks).toBe(1);

    const idx = await call(app, 'POST', '/api/knowledge/bases/product-docs/index-now');
    expect(idx.status).toBe(200);
    expect(idx.body.embedded).toBe(1);
    expect(idx.body.documentsReady).toBe(1);

    const docs = await call(app, 'GET', '/api/knowledge/bases/product-docs/documents');
    expect(docs.body.documents[0].status).toBe('ready');
    const docId = docs.body.documents[0].id;
    const doc = await call(app, 'GET', `/api/knowledge/documents/${docId}?chunks=1`);
    expect(doc.body.chunks[0].embedded).toBe(true);
    expect(doc.body.chunks[0].text).toContain('[REDACTED:email]');

    const search = await call(app, 'POST', '/api/knowledge/bases/product-docs/search', { question: 'which port?' });
    expect(search.status).toBe(200);
    expect(search.body.chunks[0].documentTitle).toBe('Ports');
    expect(search.body.graph.engine).toBe('sqlite');

    const q = await call(app, 'POST', '/api/knowledge/bases/product-docs/query', { question: 'Which port does Wall-E use?' });
    expect(q.status).toBe(200);
    expect(q.body.status).toBe('ok');
    expect(q.body.citations).toHaveLength(1);
    expect(q.body.answer).toContain('unified');

    const prov = await call(app, 'GET', `/api/knowledge/provenance/${q.body.id}`);
    expect(prov.status).toBe(200);
    expect(prov.body.actor).toMatch(/^user:/);
    expect(prov.body.retrieval[0].document.title).toBe('Ports');
    const provJson = await call(app, 'GET', `/api/knowledge/provenance/${q.body.id}/prov`);
    expect(provJson.body.entity[`walle:answer/${q.body.id}`]).toBeTruthy();
    expect((await call(app, 'GET', '/api/knowledge/provenance/nope')).status).toBe(404);

    const queries = await call(app, 'GET', '/api/knowledge/bases/product-docs/queries');
    expect(queries.body.queries[0].id).toBe(q.body.id);

    const ents = await call(app, 'GET', '/api/knowledge/bases/product-docs/graph/entities?q=wall');
    expect(ents.body.entities[0].name).toBe('Wall-E');
    const ent = await call(app, 'GET', `/api/knowledge/bases/product-docs/graph/entities/${ents.body.entities[0].id}`);
    expect(ent.body.chunkIds).toHaveLength(1);
    const gstats = await call(app, 'GET', '/api/knowledge/bases/product-docs/graph/stats');
    expect(gstats.body.entities).toBe(1);

    const audit = await call(app, 'GET', '/api/knowledge/audit?baseId=' + created.body.base.id);
    expect(audit.body.audit.map((a: any) => a.action)).toEqual(expect.arrayContaining(['base.create', 'ingest', 'query']));

    const del = await call(app, 'DELETE', `/api/knowledge/documents/${docId}`);
    expect(del.status).toBe(200);
    expect(del.body.document.status).toBe('deleted');
    const after = await call(app, 'GET', '/api/knowledge/bases/product-docs');
    expect(after.body.base.stats.documents).toBe(0);

    const delBase = await call(app, 'DELETE', '/api/knowledge/bases/product-docs');
    expect(delBase.status).toBe(200);
    expect((await call(app, 'GET', '/api/knowledge/bases/product-docs')).status).toBe(404);
  });

  it('serves ontology and governance with reload, and the cypher passthrough reports Neo4j as unavailable', async () => {
    const ont = await call(app, 'GET', '/api/knowledge/ontology');
    expect(ont.body.classes.length).toBeGreaterThan(3);
    const reload = await call(app, 'POST', '/api/knowledge/ontology/reload');
    expect(reload.status).toBe(200);
    expect(reload.body.neo4jSynced).toBe(false);
    const gov = await call(app, 'GET', '/api/knowledge/governance');
    expect(gov.body.policy.generation.require_citations).toBe(true);
    const govReload = await call(app, 'POST', '/api/knowledge/governance/reload');
    expect(govReload.body.policy.name).toBe('wall-e-default');
    const cy = await call(app, 'POST', '/api/knowledge/graph/cypher', { cypher: 'MATCH (n) RETURN n LIMIT 1' });
    expect(cy.status).toBe(503);
  });

  it('runs inline and file evals and renders the markdown report', async () => {
    await call(app, 'POST', '/api/knowledge/bases', { name: 'docs' });
    await call(app, 'POST', '/api/knowledge/bases/docs/documents', { title: 'Install', text: 'The dashboard listens on port 3001 by default.', source: 'docs/en/install/01-install.md' });
    await call(app, 'POST', '/api/knowledge/bases/docs/index-now');
    const datasets = await call(app, 'GET', '/api/knowledge/evals/datasets');
    expect(datasets.body.datasets).toContain('wall-e-docs');

    const run = await call(app, 'POST', '/api/knowledge/evals/run', {
      base: 'docs', judge: true, cases: [{ id: 'port', question: 'Which port?', expected_sources: ['01-install.md'] }],
    });
    expect(run.status).toBe(201);
    expect(run.body.run.status).toBe('done');
    expect(run.body.run.metrics.hitRate).toBe(1);
    expect(run.body.run.metrics.correctness).toBe(1);
    expect(run.body.run.results).toHaveLength(1);

    const list = await call(app, 'GET', '/api/knowledge/evals?base=docs');
    expect(list.body.runs[0].id).toBe(run.body.run.id);
    const report = await call(app, 'GET', `/api/knowledge/evals/${run.body.run.id}/report`);
    expect(report.status).toBe(200);
    expect(String(report.body)).toContain('# Knowledge eval — inline on `docs`');

    const missing = await call(app, 'POST', '/api/knowledge/evals/run', { base: 'docs', dataset: '../../etc/passwd' });
    expect(missing.status).toBe(400);
    const unknown = await call(app, 'POST', '/api/knowledge/evals/run', { base: 'docs', dataset: 'nope' });
    expect(unknown.status).toBe(404);
  });

  it('exposes RAG on /v1 with the unified key and rejects bad keys', async () => {
    await call(app, 'POST', '/api/knowledge/bases', { name: 'docs' });
    await call(app, 'POST', '/api/knowledge/bases/docs/documents', { title: 'Install', text: 'The dashboard listens on port 3001 by default.', source: 'install.md' });
    await call(app, 'POST', '/api/knowledge/bases/docs/index-now');
    const key = getUnifiedApiKey();

    expect((await call(app, 'POST', '/v1/rag/query', { knowledge_base: 'docs', question: 'port?' }, 'bad-key')).status).toBe(401);
    expect((await call(app, 'GET', '/v1/rag/bases', undefined, 'bad-key')).status).toBe(401);

    const bases = await call(app, 'GET', '/v1/rag/bases', undefined, key);
    expect(bases.body.data[0].id).toBe('docs');

    const q = await call(app, 'POST', '/v1/rag/query', { knowledge_base: 'docs', question: 'Which port?' }, key);
    expect(q.status).toBe(200);
    expect(q.body.object).toBe('knowledge.answer');
    expect(q.body.citations[0].n).toBe(1);
    expect(q.body.sources[0].text).toContain('3001');
    expect(q.body.model).toBe('test/answerer');
    expect(q.body.provenance).toBe(`/api/knowledge/provenance/${q.body.id}`);

    const s = await call(app, 'POST', '/v1/rag/search', { knowledge_base: 'docs', question: 'port' }, key);
    expect(s.body.data[0].chunk_id).toBeTruthy();

    const d = await call(app, 'POST', '/v1/rag/documents', { knowledge_base: 'docs', title: 'More', text: 'PROXY_URL points outbound requests at a proxy.' }, key);
    expect(d.status).toBe(201);
    expect(d.body.object).toBe('knowledge.document');

    expect((await call(app, 'POST', '/v1/rag/query', { knowledge_base: 'missing', question: 'x' }, key)).status).toBe(404);
  });

  it('can be disabled with KB_ENABLED=false', async () => {
    process.env.KB_ENABLED = 'false';
    expect((await call(app, 'GET', '/api/knowledge/status')).status).toBe(404);
    expect((await call(app, 'GET', '/v1/rag/bases', undefined, getUnifiedApiKey())).status).toBe(404);
  });
});
