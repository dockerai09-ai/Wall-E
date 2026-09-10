import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Express } from 'express';
import { createApp } from '../../app.js';
import { initDb, getDb, getUnifiedApiKey } from '../../db/index.js';
import { setKnowledgeChatForTests } from '../../services/knowledge/llm.js';
import { setTranscriberForTests } from '../../services/stt/pipeline.js';
import { insertRun } from '../../services/stt/store.js';
import { mintDashboardToken } from '../helpers/auth.js';

const realFetch = globalThis.fetch;

async function getJson(app: Express, path: string, token?: string) {
  const server = app.listen(0, '127.0.0.1');
  if (!server.listening) await new Promise<void>(resolve => server.once('listening', () => resolve()));
  const addr = server.address() as { port: number };
  const res = await realFetch(`http://127.0.0.1:${addr.port}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  const body = await res.json().catch(() => null);
  server.close();
  return { status: res.status, body };
}

async function post(app: Express, fields: Record<string, string>, opts: { file?: boolean; auth?: string | null } = {}) {
  const server = app.listen(0, '127.0.0.1');
  if (!server.listening) await new Promise<void>(resolve => server.once('listening', () => resolve()));
  const addr = server.address() as { port: number };
  const form = new FormData();
  if (opts.file !== false) form.append('file', new Blob([Buffer.from('RIFFfake')], { type: 'audio/wav' }), 'a.wav');
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  const headers: Record<string, string> = {};
  if (opts.auth !== null) headers.Authorization = `Bearer ${opts.auth ?? getUnifiedApiKey()}`;
  const res = await realFetch(`http://127.0.0.1:${addr.port}/v1/stt/transcriptions`, { method: 'POST', headers, body: form });
  const body = await res.json().catch(() => null);
  server.close();
  return { status: res.status, body, headers: res.headers };
}

describe('clinical STT routes', () => {
  let app: Express;
  let sttText = '';
  let chatText = '';

  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    app = createApp();
    setTranscriberForTests(async (_m, p) => ({ platform: 'groq', modelId: p.prompt ? 'whisper-large-v3-turbo' : 'whisper-large-v3', text: sttText, duration: 3 }));
    setKnowledgeChatForTests(async () => ({ text: chatText, platform: 'groq', modelId: 'fake', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }));
  });
  afterEach(() => {
    setTranscriberForTests(null);
    setKnowledgeChatForTests(null);
  });

  it('transcribes, corrects, redacts and records hash-only provenance', async () => {
    sttText = 'Jag tar metropolol 50 milligram, personnummer 85 04 12 12 30.';
    chatText = 'Jag tar metoprolol 50 mg, personnummer 85 04 12 12 30.';
    const r = await post(app, { profile: 'sv-medical', include_raw: 'true' });
    expect(r.status).toBe(200);
    expect(r.body.text).toBe('Jag tar metoprolol 50 mg, personnummer [REDACTED:personnummer].');
    expect(r.body.raw_text).toBe(sttText);
    expect(r.body.correction).toMatchObject({ model: 'groq/fake', applied: true });
    expect(r.body.redactions).toEqual([{ detector: 'personnummer', count: 1 }]);
    expect(r.body.provenance.stt.model).toBe('groq/whisper-large-v3-turbo');
    expect(r.body.provenance.id).toBe(1);
    expect(r.headers.get('x-model')).toBe('whisper-large-v3-turbo');
    const row = getDb().prepare('SELECT * FROM stt_transcripts').get() as Record<string, unknown>;
    expect(row.actor).toBe('unified');
    expect(JSON.stringify(row)).not.toContain('metoprolol');
  });

  it('omits raw text by default and honours correction=off, prompt=none, redact=false', async () => {
    sttText = 'Ring 070 123 45 67.';
    const r = await post(app, { correction: 'off', prompt: 'none', redact: 'false' });
    expect(r.status).toBe(200);
    expect(r.body.raw_text).toBeUndefined();
    expect(r.body.correction).toBeNull();
    expect(r.body.text).toBe('Ring 0701234567.');
    expect(r.body.provenance.stt.model).toBe('groq/whisper-large-v3');
    expect(r.body.provenance.prompt_hash).toBeNull();
  });

  it('validates auth, file, profile and variant fields', async () => {
    expect((await post(app, {}, { auth: null })).status).toBe(401);
    expect((await post(app, {}, { auth: 'sk-bogus' })).status).toBe(401);
    expect((await post(app, {}, { file: false })).status).toBe(400);
    expect((await post(app, { profile: 'nope' })).status).toBe(400);
    expect((await post(app, { prompt: 'weird' })).status).toBe(400);
    expect((await post(app, { temperature: '9' })).status).toBe(400);
  });

  it('serves status, runs and the glossary to the dashboard', async () => {
    const token = mintDashboardToken();
    const anon = await getJson(app, '/api/stt/status');
    expect(anon.status).toBe(401);
    insertRun({ kind: 'bench', profile: 'sv-medical', arm: 'claude-pro', loop_id: null, iteration: null, variant_json: 'null', metrics_json: JSON.stringify({ wer: 0.2 }), cases_json: '[{"id":"c01"}]', status: 'done' });
    const status = await getJson(app, '/api/stt/status', token);
    expect(status.status).toBe(200);
    expect(status.body.profiles.map((p: { name: string }) => p.name)).toContain('sv-medical');
    expect(status.body.runs[0].arm).toBe('claude-pro');
    const runs = await getJson(app, '/api/stt/runs?arm=claude-pro', token);
    expect(runs.body.runs).toHaveLength(1);
    const one = await getJson(app, '/api/stt/runs/1', token);
    expect(one.body.cases).toEqual([{ id: 'c01' }]);
    expect((await getJson(app, '/api/stt/runs/99', token)).status).toBe(404);
    const glossary = await getJson(app, '/api/stt/glossary/sv-medical', token);
    expect(glossary.body.categories['läkemedel'].length).toBeGreaterThan(5);
    expect(glossary.body.whisperPrompt).toContain('metoprolol');
    expect((await getJson(app, '/api/stt/glossary/nope', token)).status).toBe(404);
  });
});
