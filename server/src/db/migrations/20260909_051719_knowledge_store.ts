// Migration: knowledge store
// Created: 2026-09-09
//
// DOWN: reversible
//
// Tables for the knowledge module (services/knowledge/*): knowledge bases,
// documents, chunks with embeddings, an FTS5 index for hybrid retrieval, the
// ontology-constrained entity/relation graph, per-query provenance, the
// governance audit log and eval runs.
//
// SQLite is the system of record for everything, including the graph. Neo4j,
// when configured (NEO4J_URI), MIRRORS the graph tables so it can be browsed
// and queried with Cypher, and is used for traversal when reachable — but
// every feature works without it.
//
// Embeddings are stored as little-endian Float32 BLOBs, L2-normalised at write
// time so similarity is a dot product (services/knowledge/vector.ts).
//
// The FTS5 index is created best-effort: Node's built-in sqlite (the Android
// path, db/node-sqlite.ts) may lack the extension. Retrieval checks
// sqlite_master and degrades to vector + LIKE when the virtual table is missing.

import type { Db } from '../types.js';

export function up(db: Db): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS knowledge_bases (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      slug TEXT NOT NULL UNIQUE,
      name TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      -- client_profiles.id when a profile key created the base; NULL for the
      -- dashboard / unified key. Governance access.scope_by_profile uses it.
      profile_id INTEGER,
      shared INTEGER NOT NULL DEFAULT 1,
      -- Decided once at creation so every chunk lives in one vector space.
      embedder TEXT NOT NULL DEFAULT '',
      embedding_model TEXT NOT NULL DEFAULT '',
      embedding_dims INTEGER NOT NULL DEFAULT 0,
      created_at_ms INTEGER NOT NULL,
      updated_at_ms INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS knowledge_documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      title TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT '',
      content_type TEXT NOT NULL DEFAULT 'text/plain',
      content_hash TEXT NOT NULL,
      byte_size INTEGER NOT NULL,
      chunk_count INTEGER NOT NULL DEFAULT 0,
      -- indexing | ready | error | deleted (a tombstone kept for provenance)
      status TEXT NOT NULL DEFAULT 'indexing',
      error TEXT,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      redactions INTEGER NOT NULL DEFAULT 0,
      created_at_ms INTEGER NOT NULL,
      updated_at_ms INTEGER NOT NULL,
      deleted_at_ms INTEGER,
      UNIQUE(base_id, content_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_documents_base
      ON knowledge_documents(base_id, status);

    CREATE TABLE IF NOT EXISTS knowledge_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      document_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
      ordinal INTEGER NOT NULL,
      text TEXT NOT NULL,
      token_count INTEGER NOT NULL,
      char_start INTEGER NOT NULL,
      char_end INTEGER NOT NULL,
      content_hash TEXT NOT NULL,
      embedding BLOB,
      embedding_model TEXT,
      embedding_dims INTEGER,
      -- pending | done | skipped | error : graph extraction state
      graph_status TEXT NOT NULL DEFAULT 'pending',
      created_at_ms INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc
      ON knowledge_chunks(document_id, ordinal);
    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_base
      ON knowledge_chunks(base_id);
    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_unembedded
      ON knowledge_chunks(base_id) WHERE embedding IS NULL;
    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_graph_pending
      ON knowledge_chunks(base_id) WHERE graph_status = 'pending';

    CREATE TABLE IF NOT EXISTS knowledge_entities (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      class TEXT NOT NULL,
      name TEXT NOT NULL,
      -- lower-cased, whitespace-collapsed name: the merge key
      canonical TEXT NOT NULL,
      properties_json TEXT NOT NULL DEFAULT '{}',
      mention_count INTEGER NOT NULL DEFAULT 0,
      created_at_ms INTEGER NOT NULL,
      UNIQUE(base_id, class, canonical)
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_entities_canonical
      ON knowledge_entities(base_id, canonical);

    CREATE TABLE IF NOT EXISTS knowledge_mentions (
      chunk_id INTEGER NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
      entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
      confidence REAL NOT NULL DEFAULT 1,
      PRIMARY KEY (chunk_id, entity_id)
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_mentions_entity
      ON knowledge_mentions(entity_id);

    CREATE TABLE IF NOT EXISTS knowledge_relations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      from_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
      to_entity_id INTEGER NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
      type TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 1,
      -- the chunk that evidences this edge; one row per piece of evidence
      chunk_id INTEGER REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
      properties_json TEXT NOT NULL DEFAULT '{}',
      created_at_ms INTEGER NOT NULL,
      UNIQUE(from_entity_id, to_entity_id, type, chunk_id)
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_relations_from
      ON knowledge_relations(from_entity_id);
    CREATE INDEX IF NOT EXISTS idx_knowledge_relations_to
      ON knowledge_relations(to_entity_id);

    -- One row per question answered (or refused): the provenance record.
    CREATE TABLE IF NOT EXISTS knowledge_queries (
      id TEXT PRIMARY KEY,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      -- query | search | eval
      kind TEXT NOT NULL DEFAULT 'query',
      actor TEXT NOT NULL DEFAULT '',
      question TEXT NOT NULL,
      question_hash TEXT NOT NULL,
      retrieval_json TEXT NOT NULL DEFAULT '[]',
      graph_json TEXT NOT NULL DEFAULT '{}',
      prompt_hash TEXT,
      platform TEXT,
      model_id TEXT,
      answer TEXT,
      answer_hash TEXT,
      citations_json TEXT NOT NULL DEFAULT '[]',
      usage_json TEXT NOT NULL DEFAULT '{}',
      governance_json TEXT NOT NULL DEFAULT '{}',
      -- ok | refused | error
      status TEXT NOT NULL DEFAULT 'ok',
      error TEXT,
      latency_ms INTEGER NOT NULL DEFAULT 0,
      created_at_ms INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_queries_base_time
      ON knowledge_queries(base_id, created_at_ms);

    CREATE TABLE IF NOT EXISTS knowledge_audit (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      at_ms INTEGER NOT NULL,
      actor TEXT NOT NULL DEFAULT '',
      action TEXT NOT NULL,
      base_id INTEGER,
      target TEXT NOT NULL DEFAULT '',
      details_json TEXT NOT NULL DEFAULT '{}',
      policy_version TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_audit_time ON knowledge_audit(at_ms);

    CREATE TABLE IF NOT EXISTS knowledge_eval_runs (
      id TEXT PRIMARY KEY,
      base_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
      dataset TEXT NOT NULL,
      -- running | done | error
      status TEXT NOT NULL DEFAULT 'running',
      cases INTEGER NOT NULL DEFAULT 0,
      metrics_json TEXT NOT NULL DEFAULT '{}',
      results_json TEXT NOT NULL DEFAULT '[]',
      config_json TEXT NOT NULL DEFAULT '{}',
      error TEXT,
      started_at_ms INTEGER NOT NULL,
      finished_at_ms INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_knowledge_eval_runs_base
      ON knowledge_eval_runs(base_id, started_at_ms);
  `);

  // FTS5 external-content index over chunk text, kept in sync by triggers.
  // Best effort: see the header comment.
  try {
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
        text, content='knowledge_chunks', content_rowid='id', tokenize='porter unicode61'
      );
      CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_ai AFTER INSERT ON knowledge_chunks BEGIN
        INSERT INTO knowledge_chunks_fts(rowid, text) VALUES (new.id, new.text);
      END;
      CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_ad AFTER DELETE ON knowledge_chunks BEGIN
        INSERT INTO knowledge_chunks_fts(knowledge_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
      END;
      CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_au AFTER UPDATE OF text ON knowledge_chunks BEGIN
        INSERT INTO knowledge_chunks_fts(knowledge_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO knowledge_chunks_fts(rowid, text) VALUES (new.id, new.text);
      END;
    `);
  } catch {
    // No FTS5 in this SQLite build: hybrid retrieval falls back to vector + LIKE.
  }
}

export function down(db: Db): void {
  db.exec(`
    DROP TRIGGER IF EXISTS knowledge_chunks_fts_au;
    DROP TRIGGER IF EXISTS knowledge_chunks_fts_ad;
    DROP TRIGGER IF EXISTS knowledge_chunks_fts_ai;
    DROP TABLE IF EXISTS knowledge_chunks_fts;
    DROP TABLE IF EXISTS knowledge_eval_runs;
    DROP TABLE IF EXISTS knowledge_audit;
    DROP TABLE IF EXISTS knowledge_queries;
    DROP TABLE IF EXISTS knowledge_relations;
    DROP TABLE IF EXISTS knowledge_mentions;
    DROP TABLE IF EXISTS knowledge_entities;
    DROP TABLE IF EXISTS knowledge_chunks;
    DROP TABLE IF EXISTS knowledge_documents;
    DROP TABLE IF EXISTS knowledge_bases;
  `);
}
