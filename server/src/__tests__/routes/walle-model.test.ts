import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Express } from 'express';
import { createApp } from '../../app.js';
import { initDb, getDb, getUnifiedApiKey } from '../../db/index.js';
import { setKnowledgeChatForTests } from '../../services/knowledge/llm.js';
import { setTranscriberForTests } from '../../services/stt/pipeline.js';
import { extractAudioParts, loadAssistant, resolveWallEModel, renderDocument } from '../../services/stt/assistant.js';
import * as kb from '../../services/knowledge/store.js';
import { chunkText } from '../../services/knowledge/chunker.js';

const realFetch = globalThis.fetch;
const AUDIO_B64 = Buffer.from('RIFFfake').toString('base64');

async function chat(app: Express, body: unknown, opts: { auth?: string | null; raw?: boolean } = {}) {
  const server = app.listen(0, '127.0.0.1');
  if (!server.listening) await new Promise<void>(resolve => server.once('listening', () => resolve()));
  const addr = server.address() as { port: number };
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (opts.auth !== null) headers.Authorization = `Bearer ${opts.auth ?? getUnifiedApiKey()}`;
  const res = await realFetch(`http://127.0.0.1:${addr.port}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
  const text = await res.text();
  server.close();
  let json: any = null;
  if (!opts.raw) { try { json = JSON.parse(text); } catch { /* not json */ } }
  return { status: res.status, body: json, text, headers: res.headers };
}

async function models(app: Express) {
  const server = app.listen(0, '127.0.0.1');
  if (!server.listening) await new Promise<void>(resolve => server.once('listening', () => resolve()));
  const addr = server.address() as { port: number };
  const res = await realFetch(`http://127.0.0.1:${addr.port}/v1/models`, { headers: { Authorization: `Bearer ${getUnifiedApiKey()}` } });
  const json = await res.json();
  server.close();
  return json as { data: { id: string; owned_by: string; available: boolean }[] };
}

describe('wall-e/<profile> virtual model', () => {
  let app: Express;
  let sttText = '';

  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    app = createApp();
    setTranscriberForTests(async () => ({ platform: 'groq', modelId: 'whisper-large-v3', text: sttText, duration: 5 }));
    setKnowledgeChatForTests(async (messages) => {
      const system = String(messages[0]?.content ?? '');
      const user = String(messages[messages.length - 1]?.content ?? '');
      let text: string;
      if (system.includes('korrigerar')) text = user;
      else if (system.includes('strukturerar')) text = '{"kontaktorsak":"bröstsmärta","anamnes":null,"aktuella_lakemedel":[{"namn":"metoprolol","dos":"50 mg","frekvens":"1 gång dagligen"}],"allergier":["penicillin"],"symtom":["bröstsmärta"],"oklarheter":["duration oklar"]}';
      else if (system.includes('knowledge assistant')) text = 'Svaret finns i [1].';
      else if (system.includes('Wall-E')) text = `Wall-E svarar på: ${user.slice(-40)}. Ring 070 123 45 67.`;
      else text = 'okänt';
      return { text, platform: 'groq', modelId: 'fake', usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } };
    });
  });
  afterEach(() => {
    setTranscriberForTests(null);
    setKnowledgeChatForTests(null);
  });

  it('loads the assistant definition and resolves model ids', () => {
    const a = loadAssistant('sv-medical');
    expect(a.id).toBe('wall-e/sv-medical');
    expect(a.variant.sttModel).toBe('whisper-large-v3');
    expect(a.variant.structure).toBe(true);
    expect(a.persona).toContain('Wall-E');
    expect(resolveWallEModel('gpt-4o')).toBeNull();
    expect(resolveWallEModel('wall-e/sv-medical')?.profile).toBe('sv-medical');
    expect(() => resolveWallEModel('wall-e/nope')).toThrow(/Unknown model/);
  });

  it('extracts OpenAI input_audio and data-URL audio parts, refusing remote URLs', () => {
    const parts = extractAudioParts([
      { type: 'text', text: 'hej' },
      { type: 'input_audio', input_audio: { data: AUDIO_B64, format: 'mp3' } },
      { type: 'audio_url', audio_url: { url: `data:audio/wav;base64,${AUDIO_B64}` } },
    ]);
    expect(parts.map(p => [p.index, p.format, p.mimeType])).toEqual([[1, 'mp3', 'audio/mpeg'], [2, 'wav', 'audio/wav']]);
    expect(parts[0].data.toString()).toBe('RIFFfake');
    expect(() => extractAudioParts([{ type: 'audio_url', audio_url: { url: 'https://example.com/a.wav' } }])).toThrow(/remote URLs/);
  });

  it('document mode: transcribes audio, drafts journal fields, redacts and records provenance', async () => {
    sttText = 'Jag har ont i bröstet och tar metoprolol 50 milligram, mitt personnummer är 85 04 12 12 30.';
    const r = await chat(app, {
      model: 'wall-e/sv-medical',
      messages: [{ role: 'user', content: [{ type: 'input_audio', input_audio: { data: AUDIO_B64, format: 'wav' } }] }],
    });
    expect(r.status).toBe(200);
    expect(r.body.model).toBe('wall-e/sv-medical');
    expect(r.body.object).toBe('chat.completion');
    const content: string = r.body.choices[0].message.content;
    expect(content).toContain('## Transkription');
    expect(content).toContain('[REDACTED:personnummer]');
    expect(content).not.toContain('8504');
    expect(content).toContain('**Aktuella läkemedel:** metoprolol 50 mg 1 gång dagligen');
    expect(content).toContain('**Allergier:** penicillin');
    expect(r.body.wall_e.mode).toBe('document');
    expect(r.body.wall_e.transcripts[0].stt.model).toBe('groq/whisper-large-v3');
    expect(r.body.wall_e.transcripts[0].provenance_id).toBe(1);
    expect(r.body.wall_e.transcripts[0].structured.oklarheter).toEqual(['duration oklar']);
    expect(r.headers.get('x-provider')).toBe('wall-e');
    expect(getDb().prepare('SELECT COUNT(*) AS n FROM stt_transcripts').get()).toEqual({ n: 1 });
  });

  it('document mode with an instruction hands the draft to the persona, still redacted', async () => {
    sttText = 'Jag har ont i bröstet.';
    const r = await chat(app, {
      model: 'wall-e/sv-medical',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Sammanfatta i en mening.' }, { type: 'input_audio', input_audio: { data: AUDIO_B64, format: 'wav' } }] }],
    });
    expect(r.status).toBe(200);
    const content: string = r.body.choices[0].message.content;
    expect(content).toContain('## Svar');
    expect(content).toContain('Wall-E svarar på');
    // the persona's reply tried to print a phone number; governance masked it
    expect(content).toContain('[REDACTED:phone]');
    expect(r.body.wall_e.redactions).toEqual([{ detector: 'phone', count: 1 }]);
  });

  it('chat mode for text turns, and streaming emits OpenAI chunks', async () => {
    const r = await chat(app, { model: 'wall-e/sv-medical', messages: [{ role: 'system', content: 'Var extra kort.' }, { role: 'user', content: 'Vad kan du göra?' }] });
    expect(r.status).toBe(200);
    expect(r.body.wall_e.mode).toBe('chat');
    expect(r.body.choices[0].message.content).toContain('Wall-E svarar på');
    expect(r.body.usage.total_tokens).toBe(15);
    const s = await chat(app, { model: 'wall-e/sv-medical', stream: true, messages: [{ role: 'user', content: 'Hej' }] }, { raw: true });
    expect(s.status).toBe(200);
    expect(s.headers.get('content-type')).toContain('text/event-stream');
    const frames = s.text.split('\n\n').filter(l => l.startsWith('data: ')).map(l => l.slice(6));
    expect(frames[frames.length - 1]).toBe('[DONE]');
    const chunks = frames.slice(0, -1).map(f => JSON.parse(f));
    expect(chunks[0].object).toBe('chat.completion.chunk');
    expect(chunks[0].choices[0].delta.role).toBe('assistant');
    expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe('stop');
    expect(chunks.map(c => c.choices[0].delta.content ?? '').join('')).toContain('Wall-E svarar på');
  });

  it('question mode answers from the knowledge base with citations', async () => {
    const base = kb.createBase({ slug: 'rutiner', name: 'Rutiner', embedder: 'hash', embeddingModel: 'hash-256', shared: true });
    const text = 'Vid bröstsmärta tas EKG inom tio minuter.';
    kb.createDocumentWithChunks({ baseId: base.id, title: 'Bröstsmärta', source: 'rutin.md', contentType: 'text/plain', contentHash: 'h1', byteSize: text.length, chunks: chunkText(text, { targetTokens: 400, overlapTokens: 60 }) });
    const a = loadAssistant('sv-medical');
    a.knowledgeBase = base.slug;
    const { runWallEAssistant } = await import('../../services/stt/assistant.js');
    const out = await runWallEAssistant(a, { messages: [{ role: 'user', content: 'När tas EKG?' }], actor: 'test' });
    expect(out.mode).toBe('question');
    expect(out.answer?.knowledge_base).toBe(base.slug);
    expect(out.content).toContain('Svaret finns i [1]');
    expect(out.content).toContain('Källor:');
  });

  it('rejects unknown wall-e profiles, empty turns and missing auth', async () => {
    expect((await chat(app, { model: 'wall-e/nope', messages: [{ role: 'user', content: 'hej' }] })).status).toBe(400);
    expect((await chat(app, { model: 'wall-e/sv-medical', messages: [{ role: 'user', content: '' }] })).status).toBe(400);
    expect((await chat(app, { model: 'wall-e/sv-medical', messages: [{ role: 'user', content: 'hej' }] }, { auth: null })).status).toBe(401);
  });

  it('is listed in /v1/models', async () => {
    const list = await models(app);
    const entry = list.data.find(m => m.id === 'wall-e/sv-medical');
    expect(entry).toBeDefined();
    expect(entry?.owned_by).toBe('freellmapi');
    expect((entry as { name?: string } | undefined)?.name).toBe('Wall-E sv-medical');
  });

  it('renders a document with fallbacks when structuring failed', () => {
    const text = renderDocument({ part: 0, text: 'Hej.', structured: null, structure_error: 'timeout', redactions: [], correction: null, provenance_id: null, stt: { model: 'x', duration_s: null, latency_ms: 1 } }, { transcript: true, structured: true });
    expect(text).toContain('Strukturering misslyckades: timeout');
  });
});
