// Neo4j mirror of the knowledge graph. SQLite stays the system of record;
// this module pushes the same documents, chunks, entities and relations into
// Neo4j (labels from the ontology, keys derived from the SQLite ids) so the
// graph can be explored in Neo4j Browser / Bloom, queried with Cypher, and
// used for k-hop traversal at retrieval time when reachable.
//
// Everything is best effort: a missing NEO4J_URI disables the mirror, a
// connection failure is logged once per minute and never breaks ingest or
// answering. Tests inject a fake through setGraphMirrorForTests.

import neo4j, { type Driver } from 'neo4j-driver';
import { getKnowledgeConfig } from './config.js';
import type { Ontology } from './ontology.js';
import type { BaseRow, DocumentRow, ChunkRow, EntityRow, TraversalPath } from './store.js';

export interface GraphMirror {
  run(cypher: string, params?: Record<string, unknown>, mode?: 'READ' | 'WRITE'): Promise<Record<string, unknown>[]>;
  close(): Promise<void>;
}

class Neo4jMirror implements GraphMirror {
  constructor(private readonly driver: Driver, private readonly database: string) {}
  async run(cypher: string, params: Record<string, unknown> = {}, mode: 'READ' | 'WRITE' = 'WRITE'): Promise<Record<string, unknown>[]> {
    const { records } = await this.driver.executeQuery(cypher, params, {
      database: this.database,
      routing: mode === 'READ' ? neo4j.routing.READ : neo4j.routing.WRITE,
    });
    return records.map(r => r.toObject() as Record<string, unknown>);
  }
  async close(): Promise<void> {
    await this.driver.close();
  }
}

let mirror: GraphMirror | null = null;
let testMirror: GraphMirror | null | undefined;
let lastFailureAtMs = 0;
let lastError: string | null = null;
let connectedUri: string | null = null;
const RETRY_AFTER_MS = 60_000;

export function setGraphMirrorForTests(m: GraphMirror | null | undefined): void {
  testMirror = m;
}

export function isNeo4jConfigured(): boolean {
  return testMirror !== undefined ? testMirror !== null : getKnowledgeConfig().neo4j !== null;
}

function maskUri(uri: string): string {
  return uri.replace(/\/\/([^:@/]+):([^@/]+)@/, '//$1:***@');
}

/** The live mirror, or null when Neo4j is unconfigured or currently unreachable. */
export async function getGraphMirror(): Promise<GraphMirror | null> {
  if (testMirror !== undefined) return testMirror;
  const cfg = getKnowledgeConfig().neo4j;
  if (!cfg) return null;
  if (mirror && connectedUri === cfg.uri) return mirror;
  if (Date.now() - lastFailureAtMs < RETRY_AFTER_MS) return null;
  try {
    const driver = neo4j.driver(cfg.uri, neo4j.auth.basic(cfg.user, cfg.password), {
      disableLosslessIntegers: true,
      connectionTimeout: 5_000,
    });
    await driver.verifyConnectivity({ database: cfg.database });
    if (mirror) await mirror.close().catch(() => {});
    mirror = new Neo4jMirror(driver, cfg.database);
    connectedUri = cfg.uri;
    lastError = null;
    console.log(`[knowledge/neo4j] connected to ${maskUri(cfg.uri)} (db ${cfg.database})`);
    return mirror;
  } catch (err: any) {
    lastFailureAtMs = Date.now();
    lastError = String(err?.message ?? err).slice(0, 200);
    console.warn(`[knowledge/neo4j] unreachable (${maskUri(cfg.uri)}): ${lastError}; graph stays on SQLite, retrying in ${RETRY_AFTER_MS / 1000}s`);
    return null;
  }
}

export async function closeGraphMirror(): Promise<void> {
  if (mirror) await mirror.close().catch(() => {});
  mirror = null;
  connectedUri = null;
}

export function neo4jStatus(): { configured: boolean; connected: boolean; uri: string | null; error: string | null } {
  const cfg = getKnowledgeConfig().neo4j;
  return {
    configured: !!cfg || (testMirror !== undefined && testMirror !== null),
    connected: testMirror !== undefined ? testMirror !== null : (!!mirror && connectedUri === cfg?.uri),
    uri: cfg ? maskUri(cfg.uri) : null,
    error: lastError,
  };
}

// ------------------------------------------------------------------ keys ----

export const entityKey = (baseId: number, entityId: number) => `kb${baseId}:e${entityId}`;
export const chunkKey = (baseId: number, chunkId: number) => `kb${baseId}:c${chunkId}`;
export const documentKey = (baseId: number, docId: number) => `kb${baseId}:d${docId}`;

function idFromKey(key: string, kind: 'e' | 'c' | 'd'): number | null {
  const m = new RegExp(`:${kind}(\\d+)$`).exec(key);
  return m ? Number(m[1]) : null;
}

// Labels and relation types are validated identifiers (ontology.ts regexes),
// so interpolating them into Cypher is safe; values always go through params.
const SAFE_LABEL = /^[A-Z][A-Za-z0-9_]*$/;
function label(name: string): string {
  if (!SAFE_LABEL.test(name)) throw new Error(`unsafe graph label '${name}'`);
  return `\`${name}\``;
}

// -------------------------------------------------------------- ontology ----

export async function syncOntologyToNeo4j(ontology: Ontology, m?: GraphMirror | null): Promise<boolean> {
  const g = m ?? (await getGraphMirror());
  if (!g) return false;
  const constraints = [
    'CREATE CONSTRAINT knowledge_entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE',
    'CREATE CONSTRAINT knowledge_chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE c.key IS UNIQUE',
    'CREATE CONSTRAINT knowledge_document_key IF NOT EXISTS FOR (d:Document) REQUIRE d.key IS UNIQUE',
    'CREATE CONSTRAINT knowledge_base_slug IF NOT EXISTS FOR (b:KnowledgeBase) REQUIRE b.slug IS UNIQUE',
    'CREATE CONSTRAINT knowledge_ontology_class IF NOT EXISTS FOR (c:OntologyClass) REQUIRE c.name IS UNIQUE',
    'CREATE CONSTRAINT knowledge_ontology_relation IF NOT EXISTS FOR (r:OntologyRelation) REQUIRE r.name IS UNIQUE',
    'CREATE INDEX knowledge_entity_base IF NOT EXISTS FOR (e:Entity) ON (e.baseId)',
    'CREATE INDEX knowledge_entity_canonical IF NOT EXISTS FOR (e:Entity) ON (e.canonical)',
  ];
  for (const c of constraints) await g.run(c);
  await g.run(`
    MERGE (o:Ontology {name: $name})
    SET o.version = $version, o.hash = $hash, o.syncedAt = datetime(), o.description = $description
    WITH o
    UNWIND $classes AS c
      MERGE (n:OntologyClass {name: c.name})
      SET n.description = c.description, n.properties = c.properties
      MERGE (o)-[:HAS_CLASS]->(n)
    WITH DISTINCT o
    UNWIND $relations AS r
      MERGE (x:OntologyRelation {name: r.name})
      SET x.from = r.from, x.to = r.to, x.description = r.description
      MERGE (o)-[:HAS_RELATION]->(x)
      WITH x, r
      UNWIND [f IN r.from WHERE f <> 'any'] AS fromName
        MATCH (fc:OntologyClass {name: fromName}) MERGE (x)-[:FROM_CLASS]->(fc)
      WITH x, r
      UNWIND [t IN r.to WHERE t <> 'any'] AS toName
        MATCH (tc:OntologyClass {name: toName}) MERGE (x)-[:TO_CLASS]->(tc)
  `, {
    name: ontology.name,
    version: ontology.version,
    hash: ontology.hash,
    description: ontology.description ?? '',
    classes: ontology.classes.map(c => ({ name: c.name, description: c.description ?? '', properties: c.properties })),
    relations: ontology.relations.map(r => ({ name: r.name, from: r.from, to: r.to, description: r.description ?? '' })),
  });
  return true;
}

// --------------------------------------------------------------- mirror ----

export interface ChunkGraphPayload {
  base: BaseRow;
  document: DocumentRow;
  chunk: ChunkRow;
  entities: EntityRow[];
  mentions: { entityId: number; confidence: number }[];
  relations: { fromId: number; toId: number; type: string; confidence: number }[];
}

export async function mirrorChunkGraph(p: ChunkGraphPayload, m?: GraphMirror | null): Promise<boolean> {
  const g = m ?? (await getGraphMirror());
  if (!g) return false;
  const baseId = p.base.id;
  await g.run(`
    MERGE (b:KnowledgeBase {slug: $slug}) SET b.name = $baseName, b.baseId = $baseId
    MERGE (d:Document {key: $docKey})
    SET d.title = $title, d.source = $source, d.baseId = $baseId, d.documentId = $documentId, d.contentHash = $contentHash
    MERGE (b)-[:HAS_DOCUMENT]->(d)
    MERGE (c:Chunk {key: $chunkKey})
    SET c.ordinal = $ordinal, c.baseId = $baseId, c.chunkId = $chunkId, c.documentId = $documentId, c.tokenCount = $tokenCount, c.preview = $preview
    MERGE (d)-[:HAS_CHUNK]->(c)
  `, {
    slug: p.base.slug, baseName: p.base.name, baseId,
    docKey: documentKey(baseId, p.document.id), title: p.document.title, source: p.document.source, documentId: p.document.id, contentHash: p.document.content_hash,
    chunkKey: chunkKey(baseId, p.chunk.id), ordinal: p.chunk.ordinal, chunkId: p.chunk.id, tokenCount: p.chunk.token_count, preview: p.chunk.text.slice(0, 200),
  });

  const byClass = new Map<string, EntityRow[]>();
  for (const e of p.entities) {
    const list = byClass.get(e.class) ?? [];
    list.push(e);
    byClass.set(e.class, list);
  }
  for (const [cls, list] of byClass) {
    await g.run(`
      UNWIND $entities AS e
      MERGE (n:Entity {key: e.key})
      SET n.name = e.name, n.class = e.class, n.canonical = e.canonical, n.baseId = $baseId, n.entityId = e.entityId, n:${label(cls)}
    `, {
      baseId,
      entities: list.map(e => ({ key: entityKey(baseId, e.id), name: e.name, class: e.class, canonical: e.canonical, entityId: e.id })),
    });
  }
  if (p.mentions.length > 0) {
    await g.run(`
      MATCH (c:Chunk {key: $chunkKey})
      UNWIND $mentions AS m
      MATCH (n:Entity {key: m.key})
      MERGE (c)-[r:MENTIONS]->(n) SET r.confidence = m.confidence
    `, { chunkKey: chunkKey(baseId, p.chunk.id), mentions: p.mentions.map(x => ({ key: entityKey(baseId, x.entityId), confidence: x.confidence })) });
  }
  const byType = new Map<string, ChunkGraphPayload['relations']>();
  for (const r of p.relations) {
    const list = byType.get(r.type) ?? [];
    list.push(r);
    byType.set(r.type, list);
  }
  for (const [type, list] of byType) {
    await g.run(`
      UNWIND $rels AS r
      MATCH (a:Entity {key: r.from}) MATCH (b:Entity {key: r.to})
      MERGE (a)-[x:${label(type)} {chunkKey: $chunkKey}]->(b)
      SET x.confidence = r.confidence, x.baseId = $baseId
    `, { baseId, chunkKey: chunkKey(baseId, p.chunk.id), rels: list.map(r => ({ from: entityKey(baseId, r.fromId), to: entityKey(baseId, r.toId), confidence: r.confidence })) });
  }
  return true;
}

export async function removeDocumentFromNeo4j(baseId: number, documentId: number, chunkIds: number[], m?: GraphMirror | null): Promise<boolean> {
  const g = m ?? (await getGraphMirror());
  if (!g) return false;
  const chunkKeys = chunkIds.map(id => chunkKey(baseId, id));
  await g.run('MATCH ()-[r]->() WHERE r.chunkKey IN $chunkKeys DELETE r', { chunkKeys });
  await g.run('MATCH (d:Document {key: $docKey}) OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c, d', { docKey: documentKey(baseId, documentId) });
  await g.run('MATCH (n:Entity {baseId: $baseId}) WHERE NOT (n)<-[:MENTIONS]-() AND NOT (n)-[]-(:Entity) DELETE n', { baseId });
  return true;
}

export async function removeBaseFromNeo4j(baseId: number, slug: string, m?: GraphMirror | null): Promise<boolean> {
  const g = m ?? (await getGraphMirror());
  if (!g) return false;
  await g.run('MATCH (n) WHERE n.baseId = $baseId DETACH DELETE n', { baseId });
  await g.run('MATCH (b:KnowledgeBase {slug: $slug}) DETACH DELETE b', { slug });
  return true;
}

// ------------------------------------------------------------- traversal ----

export async function traverseNeo4j(
  baseId: number, seedIds: number[], maxHops: number, maxNodes = 50, m?: GraphMirror | null,
): Promise<{ nodes: number[]; edges: TraversalPath[] } | null> {
  const g = m ?? (await getGraphMirror());
  if (!g || seedIds.length === 0) return null;
  const hops = Math.max(1, Math.min(4, Math.floor(maxHops)));
  const keys = seedIds.map(id => entityKey(baseId, id));
  const nodeRows = await g.run(`
    MATCH (s:Entity) WHERE s.key IN $keys
    OPTIONAL MATCH p = (s)-[*1..${hops}]-(n:Entity {baseId: $baseId})
    WHERE NONE(r IN relationships(p) WHERE type(r) = 'MENTIONS')
    WITH collect(DISTINCT {key: s.key, depth: 0}) + collect(DISTINCT {key: n.key, depth: length(p)}) AS rows
    UNWIND rows AS row
    WITH row.key AS key, min(row.depth) AS depth WHERE key IS NOT NULL
    RETURN key, depth ORDER BY depth LIMIT $maxNodes
  `, { keys, baseId, maxNodes }, 'READ');
  const depthOf = new Map<number, number>();
  for (const row of nodeRows) {
    const id = idFromKey(String(row.key), 'e');
    if (id != null) depthOf.set(id, Number(row.depth));
  }
  const nodes = [...depthOf.keys()];
  if (nodes.length === 0) return { nodes: [], edges: [] };
  const nodeKeys = nodes.map(id => entityKey(baseId, id));
  const edgeRows = await g.run(`
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.key IN $nodeKeys AND b.key IN $nodeKeys AND type(r) <> 'MENTIONS'
    RETURN a.key AS fromKey, b.key AS toKey, type(r) AS type, r.confidence AS confidence, r.chunkKey AS chunkKey
    ORDER BY r.confidence DESC LIMIT 500
  `, { nodeKeys }, 'READ');
  const edges: TraversalPath[] = [];
  for (const row of edgeRows) {
    const fromId = idFromKey(String(row.fromKey), 'e');
    const toId = idFromKey(String(row.toKey), 'e');
    if (fromId == null || toId == null) continue;
    edges.push({
      fromId, toId, type: String(row.type), confidence: Number(row.confidence ?? 1),
      chunkId: row.chunkKey ? idFromKey(String(row.chunkKey), 'c') : null,
      depth: Math.max(depthOf.get(fromId) ?? 0, depthOf.get(toId) ?? 0),
    });
  }
  return { nodes, edges };
}

/** Read-only Cypher passthrough for the dashboard (write attempts fail in READ mode). */
export async function readCypher(cypher: string, params: Record<string, unknown> = {}, m?: GraphMirror | null): Promise<Record<string, unknown>[] | null> {
  const g = m ?? (await getGraphMirror());
  if (!g) return null;
  return g.run(cypher, params, 'READ');
}
