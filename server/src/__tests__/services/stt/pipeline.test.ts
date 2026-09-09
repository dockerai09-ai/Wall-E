import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { initDb, getDb } from '../../../db/index.js';
import { setKnowledgeChatForTests } from '../../../services/knowledge/llm.js';
import {
  transcribeClinical, setTranscriberForTests, normalizeSpokenNumbers, parseVariant, variantKey, stripWrapping, DEFAULT_VARIANT,
} from '../../../services/stt/pipeline.js';
import { MemoryCache } from '../../../services/stt/bench.js';
import { loadGlossary, whisperPrompt, correctionSystemPrompt } from '../../../services/stt/glossary.js';

const AUDIO = Buffer.from('RIFFfake');

describe('normalizeSpokenNumbers', () => {
  it('writes personnummer and phone numbers in the compact form the redactors expect', () => {
    expect(normalizeSpokenNumbers('Mitt personnummer är 85 04 12 12 30 tack')).toBe('Mitt personnummer är 850412-1230 tack');
    expect(normalizeSpokenNumbers('nummer 8504 12 12 30 och')).toBe('nummer 850412-1230 och');
    expect(normalizeSpokenNumbers('nummer 19 85 04 12 12 30 och')).toBe('nummer 19850412-1230 och');
    expect(normalizeSpokenNumbers('ring 070 123 45 67 nu')).toBe('ring 0701234567 nu');
    expect(normalizeSpokenNumbers('ring 070, 123, 45, 67 nu')).toBe('ring 0701234567 nu');
    expect(normalizeSpokenNumbers('noll sju noll ett två tre fyra fem sex sju om')).toBe('0701234567 om');
  });
  it('leaves doses, decimals and ordinary numbers alone', () => {
    expect(normalizeSpokenNumbers('blodsocker 12,5 i morse')).toBe('blodsocker 12,5 i morse');
    expect(normalizeSpokenNumbers('metoprolol 50 mg 2 gånger')).toBe('metoprolol 50 mg 2 gånger');
    expect(normalizeSpokenNumbers('vecka 28 och 3 dagar')).toBe('vecka 28 och 3 dagar');
  });
});

describe('variants and prompts', () => {
  it('parses defaults and builds a stable key', () => {
    const v = parseVariant({});
    expect(v).toEqual(DEFAULT_VARIANT);
    expect(variantKey(v)).toBe('auto|sv|prompt=glossary|t=0|corr=llm|redact=1|struct=0');
    expect(() => parseVariant({ prompt: 'bogus' })).toThrow();
  });
  it('builds a Whisper prompt within budget and a correction prompt listing the glossary', () => {
    const g = loadGlossary('sv-medical');
    const p = whisperPrompt(g);
    expect(p.length).toBeLessThanOrEqual(700);
    expect(p).toContain('metoprolol');
    expect(p.startsWith(g.context)).toBe(true);
    expect(correctionSystemPrompt(g)).toContain('läkemedel:');
  });
  it('unwraps quotes, fences and labels', () => {
    expect(stripWrapping('```\nhej\n```')).toBe('hej');
    expect(stripWrapping('"hej"')).toBe('hej');
    expect(stripWrapping('Korrigerad text: hej')).toBe('hej');
  });
});

describe('transcribeClinical', () => {
  let sttCalls = 0;
  let chatCalls = 0;
  let sttText = '';
  let chatText = '';

  beforeEach(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    sttCalls = 0;
    chatCalls = 0;
    setTranscriberForTests(async () => { sttCalls++; return { platform: 'groq', modelId: 'whisper-large-v3-turbo', text: sttText, duration: 4.2 }; });
    setKnowledgeChatForTests(async () => { chatCalls++; return { text: chatText, platform: 'groq', modelId: 'fake-70b', usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } }; });
  });
  afterEach(() => {
    setTranscriberForTests(null);
    setKnowledgeChatForTests(null);
  });

  it('applies a small correction, redacts a personnummer and persists hashes only', async () => {
    sttText = 'Jag tar metropolol 50 milligram, mitt personnummer är 85 04 12 12 30.';
    chatText = 'Jag tar metoprolol 50 mg, mitt personnummer är 85 04 12 12 30.';
    const out = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, parseVariant({}), { profile: 'sv-medical', persist: true, actor: 'test' });
    expect(out.correctionRejected).toBe(false);
    expect(out.correctedText).toContain('metoprolol');
    expect(out.finalText).toBe('Jag tar metoprolol 50 mg, mitt personnummer är [REDACTED:personnummer].');
    expect(out.redactions).toEqual([{ detector: 'personnummer', count: 1 }]);
    expect(out.stt.model).toBe('whisper-large-v3-turbo');
    expect(out.promptHash).toMatch(/^[0-9a-f]{12}$/);
    expect(out.provenanceId).toBe(1);
    const row = getDb().prepare('SELECT * FROM stt_transcripts WHERE id = 1').get() as Record<string, unknown>;
    expect(row.stt_model).toBe('whisper-large-v3-turbo');
    expect(row.correction_model).toBe('groq/fake-70b');
    expect(JSON.stringify(row)).not.toContain('metoprolol');
    expect(JSON.stringify(row)).not.toContain('8504');
    expect(row.raw_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('rejects a correction that rewrites or truncates the transcript', async () => {
    sttText = 'Mitt personnummer är 8504 12 12 30 och jag är listad på vårdcentralen i Solna.';
    chatText = 'Mitt personnummer är 8504 12 12 30 och jag är listad på vård';
    const out = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, parseVariant({}), { profile: 'sv-medical' });
    expect(out.correctionRejected).toBe(true);
    expect(out.rejectedText).toBe(chatText);
    expect(out.correctedText).toBeNull();
    expect(out.finalText).toBe('Mitt personnummer är [REDACTED:personnummer] och jag är listad på vårdcentralen i Solna.');
    chatText = 'We need to correct transcription errors: the user says something about Solna.';
    const leaked = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, parseVariant({}), { profile: 'sv-medical' });
    expect(leaked.correctionRejected).toBe(true);
  });

  it('skips the model when correction is off and keeps identifiers when redaction is off', async () => {
    sttText = 'Ring mig på 070 123 45 67.';
    const out = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, parseVariant({ correction: 'off', redact: false, prompt: 'none' }), { profile: 'sv-medical' });
    expect(chatCalls).toBe(0);
    expect(out.correction).toBeNull();
    expect(out.promptHash).toBeNull();
    expect(out.finalText).toBe('Ring mig på 0701234567.');
    expect(out.provenanceId).toBeNull();
  });

  it('answers repeated (audio, model, prompt) and (text, prompt) pairs from the cache', async () => {
    sttText = 'Jag har ont i bröstet.';
    chatText = 'Jag har ont i bröstet.';
    const cache = new MemoryCache();
    const v = parseVariant({});
    const a = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, v, { profile: 'sv-medical', cache });
    const b = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, v, { profile: 'sv-medical', cache });
    expect(a.sttCached).toBe(false);
    expect(b.sttCached).toBe(true);
    expect(sttCalls).toBe(1);
    expect(chatCalls).toBe(1);
    // a different prompt setting is a different recogniser call
    await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, { ...v, prompt: 'none' }, { profile: 'sv-medical', cache });
    expect(sttCalls).toBe(2);
    expect(chatCalls).toBe(1); // same raw text → cached correction
  });

  it('structures the final text into journal fields when asked', async () => {
    sttText = 'Jag tar metoprolol 50 mg och är allergisk mot penicillin.';
    let call = 0;
    setKnowledgeChatForTests(async () => {
      call++;
      const text = call === 1
        ? sttText
        : '{"kontaktorsak": null, "anamnes": null, "aktuella_lakemedel": [{"namn": "metoprolol", "dos": "50 mg", "frekvens": null}], "allergier": ["penicillin"], "symtom": [], "oklarheter": []}';
      return { text, platform: 'groq', modelId: 'fake', usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } };
    });
    const out = await transcribeClinical({ audio: AUDIO, filename: 'a.wav' }, parseVariant({ structure: true }), { profile: 'sv-medical' });
    expect(out.structured?.aktuella_lakemedel[0].namn).toBe('metoprolol');
    expect(out.structured?.allergier).toEqual(['penicillin']);
    expect(out.structureError).toBeNull();
  });
});
