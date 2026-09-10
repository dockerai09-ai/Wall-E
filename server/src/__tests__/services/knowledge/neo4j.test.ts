import { describe, it, expect } from 'vitest';
import { parseOntology } from '../../../services/knowledge/ontology.js';
import {
  syncOntologyToNeo4j, mirrorChunkGraph, traverseNeo4j, removeDocumentFromNeo4j, entityKey, chunkKey, type GraphMirror,
} from '../../../services/knowledge/neo4j.js';
import type { BaseRow, DocumentRow, ChunkRow, EntityRow } from '../../../services/knowledge/store.js';

class FakeMirror implements GraphMirror {
  calls: { cypher: string; params: Record<string, unknown>; mode: string }[] = [];
  responses: Record<string, unknown>[][] = [];
  async run(cypher: string, params: Record<string, unknown> = {}, mode: 'READ' | 'WRITE' = 'WRITE') {
    this.calls.push({ cypher, params, mode });
    return this.responses.shift() ?? [];
  }
  async close() {}
}

const base: BaseRow = { id: 3, slug: 'docs', name: 'Docs', description: '', profile_id: null, shared: 1, embedder: 'hash', embedding_model: 'hash-256', embedding_dims: 256, created_at_ms: 0, updated_at_ms: 0 };
const doc: DocumentRow = { id: 5, base_id: 3, title: 'T', source: 's.md', content_type: 'text/plain', content_hash: 'h', byte_size: 1, chunk_count: 1, status: 'ready', error: null, metadata_json: '{}', redactions: 0, created_at_ms: 0, updated_at_ms: 0, deleted_at_ms: null };
const chunk: ChunkRow = { id: 9, base_id: 3, document_id: 5, ordinal: 0, text: 'Wall-E uses Neo4j', token_count: 4, char_start: 0, char_end: 17, content_hash: 'c', embedding: null, embedding_model: null, embedding_dims: null, graph_status: 'done', created_at_ms: 0 };
const ent = (id: number, cls: string, name: string): EntityRow => ({ id, base_id: 3, class: cls, name, canonical: name.toLowerCase(), properties_json: '{}', mention_count: 1, created_at_ms: 0 });

describe('neo4j mirror', () => {
  it('syncs constraints and the ontology', async () => {
    const m = new FakeMirror();
    const o = parseOntology('version: 1\nname: o\nclasses:\n  - name: A\n  - name: B\nrelations:\n  - name: LINKS\n    from: [A]\n    to: [B, any]\n');
    expect(await syncOntologyToNeo4j(o, m)).toBe(true);
    expect(m.calls.filter(c => c.cypher.startsWith('CREATE CONSTRAINT')).length).toBe(6);
    const merge = m.calls.find(c => c.cypher.includes('MERGE (o:Ontology'))!;
    expect(merge.params.name).toBe('o');
    expect((merge.params.classes as any[]).map(c => c.name)).toEqual(['A', 'B']);
    expect((merge.params.relations as any[])[0].to).toEqual(['B', 'any']);
  });

  it('mirrors documents, chunks, per-class entities, mentions and typed relations', async () => {
    const m = new FakeMirror();
    const ok = await mirrorChunkGraph({
      base, document: doc, chunk,
      entities: [ent(1, 'Product', 'Wall-E'), ent(2, 'Technology', 'Neo4j'), ent(3, 'Technology', 'SQLite')],
      mentions: [{ entityId: 1, confidence: 0.9 }, { entityId: 2, confidence: 0.8 }],
      relations: [{ fromId: 1, toId: 2, type: 'USES', confidence: 0.9 }],
    }, m);
    expect(ok).toBe(true);
    const text = m.calls.map(c => c.cypher).join('\n');
    expect(text).toContain('MERGE (d:Document {key: $docKey})');
    expect(text).toContain('n:`Product`');
    expect(text).toContain('n:`Technology`');
    expect(text).toContain('MERGE (a)-[x:`USES` {chunkKey: $chunkKey}]->(b)');
    const techCall = m.calls.find(c => c.cypher.includes('n:`Technology`'))!;
    expect((techCall.params.entities as any[]).map(e => e.key)).toEqual([entityKey(3, 2), entityKey(3, 3)]);
    const mentions = m.calls.find(c => c.cypher.includes('MENTIONS'))!;
    expect(mentions.params.chunkKey).toBe(chunkKey(3, 9));
  });

  it('refuses unsafe labels', async () => {
    const m = new FakeMirror();
    await expect(mirrorChunkGraph({ base, document: doc, chunk, entities: [ent(1, 'Bad Label', 'x')], mentions: [], relations: [] }, m)).rejects.toThrow(/unsafe graph label/);
  });

  it('traverses and maps keys back to SQLite ids', async () => {
    const m = new FakeMirror();
    m.responses.push(
      [{ key: entityKey(3, 1), depth: 0 }, { key: entityKey(3, 2), depth: 1 }],
      [{ fromKey: entityKey(3, 1), toKey: entityKey(3, 2), type: 'USES', confidence: 0.9, chunkKey: chunkKey(3, 9) }],
    );
    const walk = await traverseNeo4j(3, [1], 2, 50, m);
    expect(walk).toEqual({ nodes: [1, 2], edges: [{ fromId: 1, toId: 2, type: 'USES', confidence: 0.9, chunkId: 9, depth: 1 }] });
    expect(m.calls[0].mode).toBe('READ');
    expect(m.calls[0].cypher).toContain('[*1..2]');
    expect(await traverseNeo4j(3, [], 2, 50, m)).toBeNull();
  });

  it('removes a document, its edges and orphaned entities', async () => {
    const m = new FakeMirror();
    expect(await removeDocumentFromNeo4j(3, 5, [9, 10], m)).toBe(true);
    expect(m.calls[0].params.chunkKeys).toEqual([chunkKey(3, 9), chunkKey(3, 10)]);
    expect(m.calls[1].cypher).toContain('DETACH DELETE c, d');
    expect(m.calls[2].cypher).toContain('NOT (n)<-[:MENTIONS]-()');
  });

  it('returns false without a mirror', async () => {
    expect(await mirrorChunkGraph({ base, document: doc, chunk, entities: [], mentions: [], relations: [] }, null)).toBe(false);
  });
});
