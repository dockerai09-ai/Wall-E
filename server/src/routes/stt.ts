// Clinical speech-to-text surface.
//
//   POST /v1/stt/transcriptions   API-key auth, multipart audio → governed
//                                 transcript (glossary-primed STT, LLM
//                                 correction, PII redaction, optional journal
//                                 structuring). Only hashes are persisted.
//   GET  /api/stt/status          dashboard: profiles, glossary size, run/transcript counts
//   GET  /api/stt/runs            benchmark and loop runs (metrics, no transcripts)
//   GET  /api/stt/runs/:id
//   GET  /api/stt/glossary/:profile

import { Router } from 'express';
import type { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import multer from 'multer';
import { z } from 'zod';
import { resolveAuth, type ResolvedAuth } from '../lib/system-prompt.js';
import { extractApiToken } from './proxy.js';
import { MAX_TRANSCRIPTION_BYTES, MediaError } from '../services/media.js';
import { KnowledgeError, REPO_ROOT } from '../services/knowledge/config.js';
import { transcribeClinical, parseVariant, DEFAULT_VARIANT } from '../services/stt/pipeline.js';
import { loadGlossary, glossaryStats, whisperPrompt } from '../services/stt/glossary.js';
import * as store from '../services/stt/store.js';
import { runFromRow } from '../services/stt/bench.js';

export const sttV1Router = Router();
export const sttApiRouter = Router();

function fail(res: Response, status: number, message: string, type = 'invalid_request_error'): void {
  res.status(status).json({ error: { message, type } });
}

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_TRANSCRIPTION_BYTES, files: 1 } });

const PROFILE_RE = /^[a-z0-9][a-z0-9_-]*$/i;

export function listProfiles(): string[] {
  const dir = path.join(REPO_ROOT, 'knowledge', 'stt');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(n => PROFILE_RE.test(n) && fs.existsSync(path.join(dir, n, 'glossary.yaml'))).sort();
}

const bool = (v: unknown, fallback: boolean) => {
  if (typeof v !== 'string' || v === '') return fallback;
  return !/^(false|0|no|off)$/i.test(v);
};

sttV1Router.post('/stt/transcriptions', (req: Request, res: Response, next) => {
  const auth = resolveAuth(extractApiToken(req));
  if (!auth) return fail(res, 401, 'Invalid API key', 'authentication_error');
  (req as Request & { sttAuth?: ResolvedAuth }).sttAuth = auth;
  upload.single('file')(req, res, (err: unknown) => {
    if (err) {
      if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') return fail(res, 413, `Audio file too large: the maximum upload size is ${MAX_TRANSCRIPTION_BYTES / (1024 * 1024)} MB.`);
      return fail(res, 400, 'Malformed multipart/form-data upload.');
    }
    next();
  });
}, async (req: Request, res: Response) => {
  const auth = (req as Request & { sttAuth?: ResolvedAuth }).sttAuth!;
  const file = req.file;
  if (!file || !file.buffer?.length) return fail(res, 400, 'Invalid request: `file` is required (multipart/form-data audio upload).');
  const b = req.body ?? {};
  const profile = typeof b.profile === 'string' && b.profile.trim() ? b.profile.trim() : 'sv-medical';
  if (!PROFILE_RE.test(profile) || !listProfiles().includes(profile)) return fail(res, 400, `Unknown STT profile '${profile}'. Available: ${listProfiles().join(', ') || '(none)'}.`);
  let variant;
  try {
    variant = parseVariant({
      sttModel: typeof b.model === 'string' && b.model.trim() ? b.model.trim() : DEFAULT_VARIANT.sttModel,
      language: typeof b.language === 'string' && b.language.trim() ? b.language.trim() : undefined,
      prompt: typeof b.prompt === 'string' && b.prompt ? b.prompt : DEFAULT_VARIANT.prompt,
      temperature: b.temperature !== undefined && b.temperature !== '' ? Number(b.temperature) : 0,
      correction: typeof b.correction === 'string' && b.correction ? b.correction : DEFAULT_VARIANT.correction,
      correctionModel: typeof b.correction_model === 'string' && b.correction_model.trim() ? b.correction_model.trim() : null,
      redact: bool(b.redact, true),
      structure: bool(b.structure, false),
    });
  } catch (err) {
    return fail(res, 400, err instanceof z.ZodError ? err.issues[0]?.message ?? 'invalid request' : String(err));
  }
  if (!variant.language) variant = { ...variant, language: loadGlossary(profile).language };
  const includeRaw = bool(b.include_raw, false);
  try {
    const out = await transcribeClinical(
      { audio: file.buffer, filename: file.originalname || 'audio', mimeType: file.mimetype },
      variant,
      { profile, actor: auth.kind === 'unified' ? 'unified' : `profile:${auth.profileId}`, persist: true },
    );
    res.setHeader('X-Provider', out.stt.platform);
    res.setHeader('X-Model', out.stt.model);
    res.json({
      object: 'stt.transcription',
      profile,
      text: out.finalText,
      // the un-redacted recogniser output is opt-in: it may hold identifiers
      ...(includeRaw ? { raw_text: out.rawText, corrected_text: out.correctedText } : {}),
      correction: out.correction ? { model: `${out.correction.platform}/${out.correction.model}`, applied: !out.correctionRejected, rejected: out.correctionRejected, distance: out.correctionWer, latency_ms: out.correction.latencyMs } : null,
      redactions: out.redactions,
      structured: out.structured,
      structure_error: out.structureError,
      provenance: {
        id: out.provenanceId,
        stt: { model: `${out.stt.platform}/${out.stt.model}`, duration_s: out.stt.durationS, latency_ms: out.stt.latencyMs },
        prompt_hash: out.promptHash,
        glossary_hash: out.glossaryHash,
        policy_version: out.policyVersion,
      },
      latency_ms: out.latencyMs,
    });
  } catch (err: any) {
    if (err instanceof MediaError) {
      const status = err.status >= 400 && err.status < 600 ? err.status : 502;
      return fail(res, status, `transcription error: ${err.message}`, status >= 500 ? 'api_error' : 'invalid_request_error');
    }
    if (err instanceof KnowledgeError) return fail(res, err.status, err.message, err.type);
    fail(res, 502, `transcription error: ${err?.message ?? 'unknown'}`, 'api_error');
  }
});

// ------------------------------------------------------- dashboard API ----

sttApiRouter.get('/status', (_req, res) => {
  const profiles = listProfiles().map(name => {
    try {
      const g = loadGlossary(name);
      return { name, language: g.language, glossary: glossaryStats(g), glossaryHash: g.hash, promptChars: whisperPrompt(g).length, audioSynthesized: fs.existsSync(path.join(g.dir, 'audio', 'manifest.jsonl')) };
    } catch (err: any) {
      return { name, error: String(err?.message ?? err) };
    }
  });
  res.json({ profiles, runs: store.listRuns({ limit: 5 }).map(runFromRow), transcripts: store.transcriptStats() });
});

sttApiRouter.get('/runs', (req, res) => {
  const profile = typeof req.query.profile === 'string' ? req.query.profile : undefined;
  const arm = typeof req.query.arm === 'string' ? req.query.arm : undefined;
  const loopId = typeof req.query.loop === 'string' ? req.query.loop : undefined;
  const limit = Number(req.query.limit ?? 50);
  res.json({ runs: store.listRuns({ profile, arm, loopId, limit: Number.isFinite(limit) ? limit : 50 }).map(r => ({ ...runFromRow(r), kind: r.kind, loopId: r.loop_id, iteration: r.iteration, status: r.status, profile: r.profile })) });
});

sttApiRouter.get('/runs/:id', (req, res) => {
  const row = store.getRun(Number(req.params.id));
  if (!row) return fail(res, 404, 'run not found', 'not_found');
  res.json({ ...runFromRow(row), kind: row.kind, loopId: row.loop_id, iteration: row.iteration, status: row.status, profile: row.profile, cases: JSON.parse(row.cases_json) });
});

sttApiRouter.get('/glossary/:profile', (req, res) => {
  const profile = String(req.params.profile);
  if (!PROFILE_RE.test(profile) || !listProfiles().includes(profile)) return fail(res, 404, 'profile not found', 'not_found');
  const g = loadGlossary(profile);
  res.json({ name: g.name, version: g.version, language: g.language, context: g.context, categories: g.categories, learned: g.learned, hash: g.hash, whisperPrompt: whisperPrompt(g) });
});

sttApiRouter.get('/transcripts', (req, res) => {
  const limit = Number(req.query.limit ?? 50);
  res.json({ transcripts: store.listTranscripts(Number.isFinite(limit) ? limit : 50) });
});
