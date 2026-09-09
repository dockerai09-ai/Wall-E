// Run history (stt_runs) and hash-only provenance (stt_transcripts).

import { getDb } from '../../db/index.js';

export interface SttRunRow {
  id: number;
  kind: 'bench' | 'loop';
  profile: string;
  arm: string;
  loop_id: string | null;
  iteration: number | null;
  variant_json: string;
  metrics_json: string;
  cases_json: string;
  status: string;
  created_at_ms: number;
}

export function insertRun(row: Omit<SttRunRow, 'id' | 'created_at_ms'>): SttRunRow {
  const db = getDb();
  const r = db.prepare(`
    INSERT INTO stt_runs (kind, profile, arm, loop_id, iteration, variant_json, metrics_json, cases_json, status, created_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(row.kind, row.profile, row.arm, row.loop_id, row.iteration, row.variant_json, row.metrics_json, row.cases_json, row.status, Date.now());
  return getRun(Number(r.lastInsertRowid)) as SttRunRow;
}

export function getRun(id: number): SttRunRow | null {
  return (getDb().prepare('SELECT * FROM stt_runs WHERE id = ?').get(id) as SttRunRow | undefined) ?? null;
}

export function listRuns(opts: { profile?: string; arm?: string; loopId?: string; limit?: number } = {}): SttRunRow[] {
  const where: string[] = [];
  const params: unknown[] = [];
  if (opts.profile) { where.push('profile = ?'); params.push(opts.profile); }
  if (opts.arm) { where.push('arm = ?'); params.push(opts.arm); }
  if (opts.loopId) { where.push('loop_id = ?'); params.push(opts.loopId); }
  const sql = `SELECT * FROM stt_runs ${where.length ? 'WHERE ' + where.join(' AND ') : ''} ORDER BY created_at_ms DESC, id DESC LIMIT ?`;
  params.push(Math.min(Math.max(opts.limit ?? 50, 1), 500));
  return getDb().prepare(sql).all(...params) as SttRunRow[];
}

/** Latest completed run for an arm (the loop's comparison target). */
export function latestArmRun(profile: string, arm: string): SttRunRow | null {
  return (getDb().prepare(`
    SELECT * FROM stt_runs WHERE profile = ? AND arm = ? AND status = 'done' ORDER BY created_at_ms DESC, id DESC LIMIT 1
  `).get(profile, arm) as SttRunRow | undefined) ?? null;
}

export interface TranscriptProvenance {
  profile: string;
  actor: string;
  audioSha256: string;
  audioBytes: number;
  durationS: number | null;
  sttPlatform: string;
  sttModel: string;
  promptHash: string | null;
  correctionModel: string | null;
  rawSha256: string;
  finalSha256: string;
  correctionWer: number | null;
  redactions: { detector: string; count: number }[];
  latencyMs: number;
  policyVersion: string | null;
}

export function insertTranscript(p: TranscriptProvenance): number {
  const r = getDb().prepare(`
    INSERT INTO stt_transcripts (profile, actor, audio_sha256, audio_bytes, duration_s, stt_platform, stt_model, prompt_hash,
      correction_model, raw_sha256, final_sha256, correction_wer, redactions_json, latency_ms, policy_version, created_at_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(p.profile, p.actor, p.audioSha256, p.audioBytes, p.durationS, p.sttPlatform, p.sttModel, p.promptHash,
    p.correctionModel, p.rawSha256, p.finalSha256, p.correctionWer, JSON.stringify(p.redactions), p.latencyMs, p.policyVersion, Date.now());
  return Number(r.lastInsertRowid);
}

export function listTranscripts(limit = 50): Record<string, unknown>[] {
  return getDb().prepare('SELECT * FROM stt_transcripts ORDER BY created_at_ms DESC, id DESC LIMIT ?').all(Math.min(Math.max(limit, 1), 500)) as Record<string, unknown>[];
}

export function transcriptStats(): { total: number; last24h: number } {
  const db = getDb();
  const total = (db.prepare('SELECT COUNT(*) AS n FROM stt_transcripts').get() as { n: number }).n;
  const last24h = (db.prepare('SELECT COUNT(*) AS n FROM stt_transcripts WHERE created_at_ms > ?').get(Date.now() - 86_400_000) as { n: number }).n;
  return { total, last24h };
}
