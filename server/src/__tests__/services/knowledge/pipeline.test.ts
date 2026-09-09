import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { ChatMessage } from '@freellmapi/shared/types.js';
import { initDb, getDb } from '../../../db/index.js';
import { chooseEmbedder } from '../../../services/knowledge/embedder.js';
import { ingestDocument } from '../../../services/knowledge/ingest.js';
import { runIndexerOnce } from '../../../services/knowledge/indexer.js';
import { retrieve } from '../../../services/knowledge/retrieval.js';
import { answerQuestion, REFUSAL_TEXT, parseCitations, fitContext } from '../../../services/knowledge/rag.js';
import { getProvenance, toProvJson } from '../../../services/knowledge/provenance.js';
import { setKnowledgeChatForTests } from '../../../services/knowledge/llm.js';
import { parsePolicy, setPolicyForTests, listAudit } from '../../../services/knowledge/governance.js';
import { setOntologyForTests } from '../../../services/knowledge/ontology.js';
import * as store from '../../../services/knowledge/store.js';
import { setGraphMirrorForTests } from '../../../services/knowledge/neo4j.js';
import { listTables, rebuildVirtualIndexes } from '../../../services/backups.js';

const DOC_A = `# Rotating the encryption key

Stored secrets are AES-256-GCM, so simply changing ENCRYPTION_KEY does not re-encrypt anything.
Re-encrypt them first with the rotate-encryption-key script while the server is stopped.
Contact ops@example.com if the rotation fails.

## Retention

Request analytics are retained for 90 days or 100000 request rows by default.
Set REQUEST_ANALYTICS_RETENTION_DAYS=0 to disable the age limit.`;

const DOC_B = `# Docker networking

By default the container is published only on 127.0.0.1.
Start it with HOST_BIND=0.0.0.0 docker compose up -d to reach it from the LAN.
Neo4j stores the knowledge graph; Wall-E mirrors entities into Neo4j.`;

// extractions_per_minute is maxed so one indexer pass extracts every test chunk.
const policyYaml = (extra = '') => `version: 1\nname: test\ngraph:\n  extract: true\n  extractions_per_minute: 600\n${extra}`;

function fakeChat(kind: 'answer' | 'refuse' | 'extract-only' = 'answer') {
  return async (messages: ChatMessage[]) => {
    const system = String(messages[0].content);
    const user = String(messages[1].content);
    if (system.includes('information-extraction')) {
      const entities: any[] = [];
      const relations: any[] = [];
      if (user.includes('Neo4j')) {
        entities.push({ name: 'Neo4j', class: 'Technology', confidence: 0.95 }, { name: 'Wall-E', class: 'Product', confidence: 0.95 });
        relations.push({ from: 'Wall-E', to: 'Neo4j', type: 'USES', confidence: 0.9 });
      }
      if (user.includes('AES-256-GCM')) entities.push({ name: 'AES-256-GCM', class: 'Technology', confidence: 0.9 });
      return { text: '```json\n' + JSON.stringify({ entities, relations }) + '\n```', platform: 'test', modelId: 'extractor', usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } };
    }
    if (kind === 'refuse') return { text: REFUSAL_TEXT, platform: 'test', modelId: 'answerer', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
    return { text: 'Run the rotate-encryption-key script first [1]. Analytics keep 90 days [1][2].', platform: 'test', modelId: 'answerer', usage: { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 } };
  };
}

describe('knowledge pipeline (hash embedder, SQLite graph)', () => {
  let base: store.BaseRow;

  beforeEach(async () => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    process.env.KB_EMBEDDER = 'hash';
    initDb(':memory:');
    setPolicyForTests(parsePolicy(policyYaml()));
    setOntologyForTests(null);
    setGraphMirrorForTests(null);
    setKnowledgeChatForTests(fakeChat());
    const choice = await chooseEmbedder();
    base = store.createBase({ slug: 'Docs Base', name: 'Docs', embedder: choice.embedder.kind, embeddingModel: choice.embedder.model });
  });

  afterEach(() => {
    setKnowledgeChatForTests(null);
    setPolicyForTests(null);
    setGraphMirrorForTests(undefined);
    delete process.env.KB_EMBEDDER;
  });

  it('creates a base with a slug and the hash embedder', () => {
    expect(base.slug).toBe('docs-base');
    expect(base.embedder).toBe('hash');
    expect(store.resolveBase('docs-base')?.id).toBe(base.id);
    expect(store.resolveBase(String(base.id))?.slug).toBe('docs-base');
    expect(() => store.createBase({ slug: 'docs-base', name: 'dup', embedder: 'hash', embeddingModel: 'hash-256' })).toThrow(/already exists/);
  });

  it('ingests with redaction, dedupes by content hash, and indexes to ready', async () => {
    const a = await ingestDocument(base, { title: 'Key rotation', text: DOC_A, source: 'docs/en/install/01-install.md', actor: 'user:1' });
    expect(a.deduplicated).toBe(false);
    expect(a.redactions).toBe(1);
    expect(a.document.status).toBe('indexing');
    const chunkText = store.listDocumentChunks(a.document.id).map(c => c.text).join(' ');
    expect(chunkText).toContain('[REDACTED:email]');
    expect(chunkText).not.toContain('ops@example.com');

    const again = await ingestDocument(base, { title: 'Key rotation copy', text: DOC_A, actor: 'user:1' });
    expect(again.deduplicated).toBe(true);
    expect(again.document.id).toBe(a.document.id);

    await ingestDocument(base, { title: 'Networking', text: DOC_B, source: 'docs/en/install/01-install.md#docker', actor: 'user:1' });

    const report = await runIndexerOnce();
    expect(report.embedded).toBeGreaterThan(0);
    expect(report.extracted).toBeGreaterThan(0);
    expect(report.extractionErrors).toBe(0);
    const stats = store.baseStats(base.id);
    expect(stats.documents).toBe(2);
    expect(stats.documentsReady).toBe(2);
    expect(stats.chunksEmbedded).toBe(stats.chunks);
    expect(stats.chunksGraphPending).toBe(0);
    expect(stats.entities).toBeGreaterThanOrEqual(3);
    expect(stats.relations).toBe(1);
    expect(store.getBase(base.id)!.embedding_dims).toBe(256);

    const actions = listAudit({ baseId: base.id }).map(r => r.action);
    expect(actions).toContain('ingest');
    expect(actions).toContain('ingest.deduplicated');
  });

  it('retrieves with hybrid fusion and graph expansion, then answers with citations and provenance', async () => {
    await ingestDocument(base, { title: 'Key rotation', text: DOC_A, source: 'docs/en/install/01-install.md', actor: 'user:1' });
    await ingestDocument(base, { title: 'Networking', text: DOC_B, source: 'docs/en/install/01-install.md#docker', actor: 'user:1' });
    await runIndexerOnce();

    const r = await retrieve(base, 'How do I rotate the ENCRYPTION_KEY?');
    expect(r.fts).toBe(true);
    expect(r.chunks.length).toBeGreaterThan(0);
    expect(r.chunks[0].score).toBe(1);
    expect(r.chunks[0].text).toContain('rotate-encryption-key');
    expect(r.chunks[0].via).toContain('keyword');
    expect(r.embedder.kind).toBe('hash');

    const g = await retrieve(base, 'What does Wall-E use Neo4j for?');
    expect(g.graph.engine).toBe('sqlite');
    expect(g.graph.seeds.map(n => n.name)).toEqual(expect.arrayContaining(['Neo4j']));
    expect(g.graph.nodes.map(n => n.name)).toEqual(expect.arrayContaining(['Neo4j', 'Wall-E']));
    expect(g.graph.edges.some(e => e.type === 'USES')).toBe(true);
    expect(g.chunks.some(c => c.via.includes('graph'))).toBe(true);

    const a = await answerQuestion(base, 'How do I rotate the key?', { actor: 'user:1' });
    expect(a.status).toBe('ok');
    expect(a.citations.map(c => c.n)).toEqual([1, 2]);
    expect(a.model).toEqual({ platform: 'test', modelId: 'answerer' });
    expect(a.governance.citationsMissing).toBe(false);
    expect(a.governance.policyVersion).toMatch(/^test@1#/);
    expect(a.governance.embedder).toBe('hash-256');

    const prov = getProvenance(a.id)!;
    expect(prov.status).toBe('ok');
    expect(prov.question).toBe('How do I rotate the key?');
    expect(prov.retrieval.length).toBe(a.sources.length);
    expect(prov.retrieval[0].document?.title).toBeTruthy();
    expect(prov.citations).toHaveLength(2);
    expect(prov.promptHash).toMatch(/^[0-9a-f]{64}$/);
    const provJson = toProvJson(prov) as any;
    expect(provJson.entity[`walle:answer/${a.id}`]).toBeTruthy();
    expect(provJson.agent['walle:model/test/answerer']['prov:type']).toBe('prov:SoftwareAgent');
    expect(Object.keys(provJson.wasDerivedFrom).length).toBeGreaterThan(2);
    expect(store.listQueries(base.id)[0].id).toBe(a.id);
  });

  it('refuses without evidence and records the refusal', async () => {
    const a = await answerQuestion(base, 'Anything at all?', { actor: 'unified' });
    expect(a.status).toBe('refused');
    expect(a.answer).toBe(REFUSAL_TEXT);
    expect(a.model).toBeNull();
    expect(getProvenance(a.id)!.status).toBe('refused');
    expect(listAudit({ action: 'query.refused' })).toHaveLength(1);

    setPolicyForTests(parsePolicy(policyYaml('generation:\n  refuse_without_evidence: false\n')));
    setKnowledgeChatForTests(fakeChat('refuse'));
    const b = await answerQuestion(base, 'Anything?', { actor: 'unified' });
    expect(b.status).toBe('refused');
    expect(b.model?.modelId).toBe('answerer');
  });

  it('flags missing citations and records model errors', async () => {
    await ingestDocument(base, { title: 'A', text: DOC_A, actor: 'u' });
    await runIndexerOnce();
    setKnowledgeChatForTests(async () => ({ text: 'No brackets here.', platform: 'test', modelId: 'm', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }));
    const a = await answerQuestion(base, 'rotate key', { actor: 'u' });
    expect(a.status).toBe('ok');
    expect(a.citations).toEqual([]);
    expect(a.governance.citationsMissing).toBe(true);

    setKnowledgeChatForTests(async () => { throw new Error('provider exploded'); });
    await expect(answerQuestion(base, 'rotate key', { actor: 'u' })).rejects.toThrow(/provider exploded/);
    const errored = store.listQueries(base.id).find(q => q.status === 'error');
    expect(errored?.error).toContain('provider exploded');
  });

  it('deletes a document with a tombstone, cascades chunks/graph, and provenance survives', async () => {
    const a = await ingestDocument(base, { title: 'Networking', text: DOC_B, source: 'net.md', actor: 'u' });
    await runIndexerOnce();
    const ans = await answerQuestion(base, 'Neo4j', { actor: 'u' });
    expect(ans.sources.length).toBeGreaterThan(0);

    const tomb = store.deleteDocument(a.document.id)!;
    expect(tomb.status).toBe('deleted');
    expect(tomb.deleted_at_ms).not.toBeNull();
    expect(store.listDocumentChunks(a.document.id)).toHaveLength(0);
    expect(store.baseStats(base.id).entities).toBe(0);
    expect(store.listDocuments(base.id)).toHaveLength(0);
    expect(store.listDocuments(base.id, { includeDeleted: true })).toHaveLength(1);

    const prov = getProvenance(ans.id)!;
    expect(prov.retrieval[0].document?.status).toBe('deleted');
    expect(prov.retrieval[0].chunk).toBeNull();

    // Re-ingesting identical content after deletion is allowed.
    const again = await ingestDocument(base, { title: 'Networking', text: DOC_B, source: 'net.md', actor: 'u' });
    expect(again.deduplicated).toBe(false);
  });

  it('enforces content type and size limits from the policy', async () => {
    await expect(ingestDocument(base, { title: 'x', text: 'hello', contentType: 'application/pdf', actor: 'u' })).rejects.toMatchObject({ status: 415 });
    setPolicyForTests(parsePolicy(policyYaml('ingest:\n  max_document_bytes: 10\n')));
    await expect(ingestDocument(base, { title: 'x', text: 'this is more than ten bytes', actor: 'u' })).rejects.toMatchObject({ status: 413 });
    expect(listAudit({ action: 'ingest.rejected' })).toHaveLength(2);
  });

  it('reindexes after clearing embeddings', async () => {
    await ingestDocument(base, { title: 'A', text: DOC_A, actor: 'u' });
    await runIndexerOnce();
    expect(store.baseStats(base.id).chunksEmbedded).toBeGreaterThan(0);
    store.clearEmbeddings(base.id);
    expect(store.baseStats(base.id).chunksEmbedded).toBe(0);
    expect(store.basesWithPendingWork()).toContain(base.id);
    await runIndexerOnce();
    expect(store.baseStats(base.id).chunksGraphPending).toBe(0);
  });

  it('keeps the FTS5 index and its shadow tables out of backups and can rebuild it', async () => {
    await ingestDocument(base, { title: 'A', text: DOC_A, actor: 'u' });
    const tables = listTables();
    expect(tables).toContain('knowledge_chunks');
    expect(tables).toContain('knowledge_documents');
    expect(tables.some(t => t.startsWith('knowledge_chunks_fts'))).toBe(false);
    expect(rebuildVirtualIndexes()).toEqual(['knowledge_chunks_fts']);
    const r = await retrieve(base, 'rotate the encryption key', { graph: false });
    expect(r.chunks.length).toBeGreaterThan(0);
  });

  it('retries transient extraction failures before marking a chunk as error', async () => {
    await ingestDocument(base, { title: 'B', text: DOC_B, actor: 'u' });
    let calls = 0;
    setKnowledgeChatForTests(async () => { calls++; throw new Error('429 rate limited'); });
    await runIndexerOnce();
    expect(calls).toBeGreaterThan(0);
    // First miss: back to pending, not error.
    expect((getDb().prepare("SELECT COUNT(*) n FROM knowledge_chunks WHERE graph_status = 'error'").get() as any).n).toBe(0);
    expect(store.baseStats(base.id).chunksGraphPending).toBeGreaterThan(0);
    await runIndexerOnce();
    await runIndexerOnce();
    expect(store.baseStats(base.id).chunksGraphPending).toBe(0);
    expect((getDb().prepare("SELECT COUNT(*) n FROM knowledge_chunks WHERE graph_status = 'error'").get() as any).n).toBeGreaterThan(0);
    expect(store.baseStats(base.id).documentsReady).toBe(1);
    // graph-only reindex re-queues the failures without touching embeddings.
    expect(store.resetGraphState(base.id)).toBeGreaterThan(0);
    expect(store.baseStats(base.id).chunksEmbedded).toBe(store.baseStats(base.id).chunks);
    setKnowledgeChatForTests(fakeChat());
    await runIndexerOnce();
    expect(store.baseStats(base.id).chunksGraphPending).toBe(0);
    expect(store.baseStats(base.id).entities).toBeGreaterThan(0);
  });

  it('skips extraction when the policy disables it', async () => {
    setPolicyForTests(parsePolicy('version: 1\nname: t\ngraph:\n  extract: false\n'));
    await ingestDocument(base, { title: 'B', text: DOC_B, actor: 'u' });
    const r = await runIndexerOnce();
    expect(r.extracted).toBe(0);
    expect(store.baseStats(base.id).entities).toBe(0);
    expect(store.baseStats(base.id).documentsReady).toBe(1);
    expect((getDb().prepare("SELECT COUNT(*) n FROM knowledge_chunks WHERE graph_status = 'skipped'").get() as any).n).toBeGreaterThan(0);
  });
});

describe('rag helpers', () => {
  const src = (i: number) => ({ chunkId: i, documentId: 1, documentTitle: 't', source: 's', ordinal: i, text: 'x'.repeat(400), score: 1, vectorRank: null, keywordRank: null, via: [] as any });

  it('parses citation groups and ignores out-of-range numbers', () => {
    const sources = [src(1), src(2), src(3)];
    expect(parseCitations('A [1] B [2, 3] C [9] D [3][1]', sources).map(c => c.n)).toEqual([1, 2, 3]);
    expect(parseCitations('nothing', sources)).toEqual([]);
  });

  it('fits sources into the context budget, always keeping the first', () => {
    const sources = [src(1), src(2), src(3), src(4)];
    expect(fitContext(sources, 250)).toHaveLength(2);
    expect(fitContext(sources, 10)).toHaveLength(1);
    expect(fitContext(sources, 10_000)).toHaveLength(4);
  });
});
