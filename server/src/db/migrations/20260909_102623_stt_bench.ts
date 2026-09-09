// Migration: stt_bench
// Created: 2026-09-09
//
// Clinical speech-to-text: benchmark/loop run history and per-request
// provenance. Transcript TEXT is never stored here — patient speech is
// protected health data — only content hashes, model identities and
// redaction counts, enough to prove what produced an output.
//
// DOWN: reversible

import type { Db } from '../types.js';

export function up(db: Db): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS stt_runs (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      kind          TEXT NOT NULL,                -- 'bench' | 'loop'
      profile       TEXT NOT NULL,                -- e.g. 'sv-medical'
      arm           TEXT NOT NULL,                -- 'wall-e' | 'claude-pro' | 'generic-whisper' | ...
      loop_id       TEXT,                         -- groups the iterations of one loop run
      iteration     INTEGER,
      variant_json  TEXT NOT NULL,                -- pipeline configuration that produced the run
      metrics_json  TEXT NOT NULL,                -- aggregate metrics
      cases_json    TEXT NOT NULL,                -- per-case scores (references and hypotheses excluded)
      status        TEXT NOT NULL DEFAULT 'done', -- 'done' | 'error'
      created_at_ms INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_stt_runs_profile ON stt_runs(profile, created_at_ms DESC);
    CREATE INDEX IF NOT EXISTS idx_stt_runs_loop ON stt_runs(loop_id, iteration);

    CREATE TABLE IF NOT EXISTS stt_transcripts (
      id               INTEGER PRIMARY KEY AUTOINCREMENT,
      profile          TEXT NOT NULL,
      actor            TEXT NOT NULL DEFAULT '',
      audio_sha256     TEXT NOT NULL,
      audio_bytes      INTEGER NOT NULL,
      duration_s       REAL,
      stt_platform     TEXT NOT NULL,
      stt_model        TEXT NOT NULL,
      prompt_hash      TEXT,
      correction_model TEXT,
      raw_sha256       TEXT NOT NULL,
      final_sha256     TEXT NOT NULL,
      correction_wer   REAL,                      -- distance raw -> corrected (hallucination guard input)
      redactions_json  TEXT NOT NULL DEFAULT '[]',
      latency_ms       INTEGER NOT NULL,
      policy_version   TEXT,
      created_at_ms    INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_stt_transcripts_created ON stt_transcripts(created_at_ms DESC);
  `);
}

export function down(db: Db): void {
  db.exec(`
    DROP INDEX IF EXISTS idx_stt_transcripts_created;
    DROP TABLE IF EXISTS stt_transcripts;
    DROP INDEX IF EXISTS idx_stt_runs_loop;
    DROP INDEX IF EXISTS idx_stt_runs_profile;
    DROP TABLE IF EXISTS stt_runs;
  `);
}
