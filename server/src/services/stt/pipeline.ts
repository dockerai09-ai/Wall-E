// Clinical speech-to-text pipeline (Swedish first, profile-driven):
//
//   audio ─► STT (Wall-E's /v1/audio/transcriptions chain: Groq/Cloudflare
//            Whisper or a registered local endpoint such as KB-Whisper), primed
//            with the profile glossary as the Whisper prompt
//         ─► spoken-number normalisation ("noll sju noll" → 070, digit groups)
//         ─► optional LLM correction pass (governed model, temperature 0,
//            glossary in the prompt, hallucination guard: a "correction" that
//            rewrites more than a third of the words is discarded)
//         ─► PII redaction with the knowledge governance policy (personnummer,
//            phone, email…) — on by default, this is patient speech
//         ─► optional structuring into journal fields (JSON, validated)
//         ─► hash-only provenance row (no transcript text is persisted).
//
// Every knob is a field of SttVariant so the benchmark and the loop can
// explore the space with the same code path the API serves.

import crypto from 'crypto';
import { z } from 'zod';
import { runTranscription, type TranscriptionResult } from '../media.js';
import { knowledgeChat, extractJsonObject, type ChatResult } from '../knowledge/llm.js';
import { getPolicy, redactText, luhnOk } from '../knowledge/governance.js';
import { loadGlossary, whisperPrompt, correctionSystemPrompt, type Glossary } from './glossary.js';
import { wer, normalizeSwedish, parseSwedishNumber } from './metrics.js';
import { insertTranscript } from './store.js';

export interface SttVariant {
  /** 'auto' (router picks) or a provider model id such as 'whisper-large-v3'. */
  sttModel: string;
  language: string;
  prompt: 'none' | 'glossary';
  temperature: number;
  correction: 'off' | 'llm';
  /** Pinned catalog model "platform/model_id" or null for the router chain. */
  correctionModel: string | null;
  redact: boolean;
  structure: boolean;
}

export const DEFAULT_VARIANT: SttVariant = {
  sttModel: 'auto',
  language: 'sv',
  prompt: 'glossary',
  temperature: 0,
  correction: 'llm',
  correctionModel: null,
  redact: true,
  structure: false,
};

const variantSchema = z.object({
  sttModel: z.string().min(1).default(DEFAULT_VARIANT.sttModel),
  language: z.string().min(2).max(8).default(DEFAULT_VARIANT.language),
  prompt: z.enum(['none', 'glossary']).default(DEFAULT_VARIANT.prompt),
  temperature: z.number().min(0).max(1).default(0),
  correction: z.enum(['off', 'llm']).default(DEFAULT_VARIANT.correction),
  correctionModel: z.string().min(1).nullable().default(null),
  redact: z.boolean().default(true),
  structure: z.boolean().default(false),
});

export function parseVariant(raw: unknown): SttVariant {
  return variantSchema.parse(raw ?? {});
}

/** Stable identity of a variant (used for cache keys and run labels). */
export function variantKey(v: SttVariant): string {
  return [
    v.sttModel, v.language, `prompt=${v.prompt}`, `t=${v.temperature}`,
    `corr=${v.correction}${v.correctionModel ? ':' + v.correctionModel : ''}`,
    `redact=${v.redact ? 1 : 0}`, `struct=${v.structure ? 1 : 0}`,
  ].join('|');
}

export interface SttInput {
  audio: Buffer;
  filename: string;
  mimeType?: string;
}

export interface StructuredNote {
  kontaktorsak: string | null;
  anamnes: string | null;
  aktuella_lakemedel: { namn: string; dos: string | null; frekvens: string | null }[];
  allergier: string[];
  symtom: string[];
  oklarheter: string[];
}

const structuredSchema = z.object({
  kontaktorsak: z.string().nullable().default(null),
  anamnes: z.string().nullable().default(null),
  aktuella_lakemedel: z.array(z.object({
    namn: z.string(), dos: z.string().nullable().default(null), frekvens: z.string().nullable().default(null),
  })).default([]),
  allergier: z.array(z.string()).default([]),
  symtom: z.array(z.string()).default([]),
  oklarheter: z.array(z.string()).default([]),
});

export interface SttOutput {
  rawText: string;
  correctedText: string | null;
  correctionRejected: boolean;
  /** What the model proposed when the guard rejected it (diagnostics). */
  rejectedText: string | null;
  correctionWer: number | null;
  finalText: string;
  redactions: { detector: string; count: number }[];
  structured: StructuredNote | null;
  structureError: string | null;
  stt: { platform: string; model: string; durationS: number | null; latencyMs: number };
  correction: { platform: string; model: string; latencyMs: number; usage: ChatResult['usage'] } | null;
  promptHash: string | null;
  glossaryHash: string;
  policyVersion: string;
  latencyMs: number;
  provenanceId: number | null;
  /** True when the recogniser output came from the benchmark cache. */
  sttCached: boolean;
}

/** Optional memo for the benchmark: identical (audio, model, prompt) and
 *  (raw text, correction prompt, model) pairs are answered from the cache so a
 *  loop that changes one knob does not re-pay for the others. */
export interface SttCache {
  get(key: string): unknown | null;
  set(key: string, value: unknown): void;
}

export interface SttRunOptions {
  profile: string;
  actor?: string;
  glossary?: Glossary;
  /** Write a hash-only provenance row (the API does; the benchmark does not). */
  persist?: boolean;
  waitForResetSeconds?: number;
  cache?: SttCache;
  /** Awaited before every real provider call (rate limiting). */
  throttle?: () => Promise<void>;
}

/** Share of words the correction may change before it is treated as a rewrite. */
export const CORRECTION_MAX_WER = 0.25;
export const CORRECTION_MAX_TOKENS = 2048;
/** The corrected text must keep between these fractions of the raw word count. */
export const CORRECTION_LENGTH_RATIO: [number, number] = [0.85, 1.25];

function looksLikePersonnummer(digits: string): boolean {
  const ten = digits.slice(-10);
  const month = Number(ten.slice(2, 4));
  const day = Number(ten.slice(4, 6));
  return month >= 1 && month <= 12 && day >= 1 && day <= 91 && luhnOk(ten);
}

type Transcriber = (model: string, p: {
  file: Buffer; filename: string; mimeType?: string; language?: string; prompt?: string; temperature?: number;
}) => Promise<Pick<TranscriptionResult, 'platform' | 'modelId' | 'text' | 'duration'>>;

let transcriberOverride: Transcriber | null = null;

/** Tests inject a fake recogniser so nothing here needs a provider key. */
export function setTranscriberForTests(fn: Transcriber | null): void {
  transcriberOverride = fn;
}

async function transcribe(model: string, p: Parameters<Transcriber>[1]) {
  if (transcriberOverride) return transcriberOverride(model, p);
  return runTranscription(model, { ...p, responseFormat: 'verbose_json' });
}

const sha = (s: string | Buffer) => crypto.createHash('sha256').update(s).digest('hex');

// --------------------------------------------------- spoken numbers -------

const SMALL_NUMBER_WORD = /^(noll|en|ett|två|tre|fyra|fem|sex|sju|åtta|nio|tio|elva|tolv|tretton|fjorton|femton|sexton|sjutton|arton|nitton|tjugo|trettio|fyrtio|femtio|sextio|sjuttio|åttio|nittio|tjugo\w+|trettio\w+|fyrtio\w+|femtio\w+|sextio\w+|sjuttio\w+|åttio\w+|nittio\w+)$/i;

/** Make identifiers a recogniser wrote as spoken groups look like the written
 *  forms the redaction regexes expect: "85 04 12 12 34" → "850412-1234",
 *  "070, 123, 45, 67" → "0701234567", "noll sju noll ett två tre fyra fem sex
 *  sju" → "0701234567". Ordinary prose is left alone. */
export function normalizeSpokenNumbers(text: string): string {
  let t = text;
  // runs of ≥ 4 small number words → digits ("noll sju noll ett två…")
  const parts = t.split(/(\s+)/);
  const isNumberWord = (w: string) => SMALL_NUMBER_WORD.test(w.replace(/[,.:;]+$/, ''));
  for (let i = 0; i < parts.length; i++) {
    if (!parts[i] || /^\s+$/.test(parts[i]) || !isNumberWord(parts[i])) continue;
    let j = i;
    const words: string[] = [];
    while (j < parts.length && (/^\s+$/.test(parts[j]) || isNumberWord(parts[j]))) {
      if (!/^\s+$/.test(parts[j])) words.push(parts[j]);
      j++;
    }
    // trailing whitespace belongs to the text after the run
    if (j > i && /^\s+$/.test(parts[j - 1])) j--;
    if (words.length >= 4) {
      const trailing = /[,.:;]+$/.exec(words[words.length - 1])?.[0] ?? '';
      const digits = words.map(w => parseSwedishNumber(w.replace(/[,.:;]+$/, '')));
      if (digits.every((n): n is number => n != null)) {
        parts.splice(i, j - i, digits.join('') + trailing);
      }
    }
    i = Math.max(i, i + 1);
  }
  t = parts.join('');
  // commas between digit GROUPS → spaces ("070, 123, 45"); a decimal comma
  // ("12,5") has a single digit after it and is left alone
  t = t.replace(/(\d{2,})\s*,\s*(?=\d{2})/g, '$1 ');
  // Runs of digit groups ("85 04 12 12 30", "8504 12 12 30", "070 123 45 67"):
  // concatenate and, when the digits have the shape of a personnummer or a
  // Swedish phone number, write the compact form the redactors recognise.
  t = t.replace(/\b\d{1,4}(?:[\s-]+\d{1,4}){1,}\b/g, run => {
    const digits = run.replace(/\D/g, '');
    // "070 123 45 67": a leading three-digit group starting with 0 is how a
    // Swedish phone number is read; a personnummer is read in pairs or 6+4
    const phoneShape = /^0\d{2}[\s-]/.test(run) && /^0\d{7,9}$/.test(digits);
    if (!phoneShape && (digits.length === 10 || digits.length === 12) && looksLikePersonnummer(digits)) return `${digits.slice(0, -4)}-${digits.slice(-4)}`;
    if (/^0\d{7,9}$/.test(digits)) return digits;
    return run;
  });
  return t;
}

// ---------------------------------------------------------- pipeline ------

export async function transcribeClinical(input: SttInput, variant: SttVariant, opts: SttRunOptions): Promise<SttOutput> {
  const started = Date.now();
  const glossary = opts.glossary ?? loadGlossary(opts.profile);
  const policy = getPolicy();
  const prompt = variant.prompt === 'glossary' ? whisperPrompt(glossary) : undefined;

  const sttStarted = Date.now();
  const audioSha = sha(input.audio);
  const sttCacheKey = `stt:${sha([audioSha, variant.sttModel, variant.language, prompt ?? '', variant.temperature].join('\u0001'))}`;
  let stt = opts.cache?.get(sttCacheKey) as Awaited<ReturnType<typeof transcribe>> | null;
  let sttCached = !!stt;
  if (!stt) {
    if (opts.throttle) await opts.throttle();
    stt = await transcribe(variant.sttModel, {
      file: input.audio, filename: input.filename, mimeType: input.mimeType,
      language: variant.language || undefined, prompt, temperature: variant.temperature,
    });
    opts.cache?.set(sttCacheKey, { platform: stt.platform, modelId: stt.modelId, text: stt.text, duration: stt.duration ?? null });
    sttCached = false;
  }
  const sttLatency = Date.now() - sttStarted;
  const rawText = stt.text.trim();

  let correctedText: string | null = null;
  let correctionRejected = false;
  let correctionWer: number | null = null;
  let rejectedText: string | null = null;
  let correction: SttOutput['correction'] = null;
  if (variant.correction === 'llm' && rawText) {
    const cStarted = Date.now();
    const system = correctionSystemPrompt(glossary);
    const corrKey = `corr:${sha([sha(rawText), sha(system), variant.correctionModel ?? '', String(CORRECTION_MAX_TOKENS)].join('\u0001'))}`;
    let r = opts.cache?.get(corrKey) as ChatResult | null;
    if (!r) {
      if (opts.throttle) await opts.throttle();
      r = await knowledgeChat([
        { role: 'system', content: system },
        { role: 'user', content: rawText },
      ], {
        temperature: 0,
        // reasoning models (gpt-oss) think before they answer and the thinking
        // counts against max_tokens: a tight budget returns truncated reasoning
        // instead of the corrected text
        maxTokens: CORRECTION_MAX_TOKENS,
        model: variant.correctionModel,
        timeoutMs: 60_000,
        waitForResetSeconds: opts.waitForResetSeconds ?? policy.generation.wait_for_reset_seconds,
      });
      opts.cache?.set(corrKey, r);
    }
    const candidate = stripWrapping(r.text);
    correctionWer = wer(rawText, candidate);
    const ratio = normalizeSwedish(candidate).length / Math.max(1, normalizeSwedish(rawText).length);
    if (!candidate || correctionWer > CORRECTION_MAX_WER || ratio < CORRECTION_LENGTH_RATIO[0] || ratio > CORRECTION_LENGTH_RATIO[1]) {
      correctionRejected = true;
      rejectedText = candidate;
    } else {
      correctedText = candidate;
    }
    correction = { platform: r.platform, model: r.modelId, latencyMs: Date.now() - cStarted, usage: r.usage };
  }

  let finalText = normalizeSpokenNumbers(correctedText ?? rawText);
  let redactions: SttOutput['redactions'] = [];
  if (variant.redact) {
    const report = redactText(finalText, { ...policy, ingest: { ...policy.ingest, redact_pii: true } });
    finalText = report.text;
    redactions = report.redactions;
  }

  let structured: StructuredNote | null = null;
  let structureError: string | null = null;
  if (variant.structure && finalText) {
    try {
      const r = await knowledgeChat([
        { role: 'system', content: STRUCTURE_PROMPT },
        { role: 'user', content: finalText },
      ], { temperature: 0, maxTokens: 1024, model: variant.correctionModel, timeoutMs: 60_000, waitForResetSeconds: opts.waitForResetSeconds ?? policy.generation.wait_for_reset_seconds });
      const parsed = structuredSchema.safeParse(extractJsonObject(r.text));
      if (parsed.success) structured = parsed.data;
      else structureError = 'model returned no valid journal JSON';
    } catch (err: any) {
      structureError = String(err?.message ?? err);
    }
  }

  const latencyMs = Date.now() - started;
  let provenanceId: number | null = null;
  if (opts.persist) {
    try {
      provenanceId = insertTranscript({
        profile: opts.profile,
        actor: opts.actor ?? '',
        audioSha256: audioSha,
        audioBytes: input.audio.length,
        durationS: typeof stt.duration === 'number' ? stt.duration : null,
        sttPlatform: stt.platform,
        sttModel: stt.modelId,
        promptHash: prompt ? sha(prompt).slice(0, 12) : null,
        correctionModel: correction ? `${correction.platform}/${correction.model}` : null,
        rawSha256: sha(rawText),
        finalSha256: sha(finalText),
        correctionWer,
        redactions,
        latencyMs,
        policyVersion: policy.versionTag,
      });
    } catch (err: any) {
      console.error(`[stt] provenance row failed: ${err?.message ?? err}`);
    }
  }

  return {
    rawText,
    correctedText,
    correctionRejected,
    rejectedText,
    correctionWer,
    finalText,
    redactions,
    structured,
    structureError,
    stt: { platform: stt.platform, model: stt.modelId, durationS: typeof stt.duration === 'number' ? stt.duration : null, latencyMs: sttLatency },
    correction,
    promptHash: prompt ? sha(prompt).slice(0, 12) : null,
    glossaryHash: glossary.hash,
    policyVersion: policy.versionTag,
    latencyMs,
    provenanceId,
    sttCached,
  };
}

/** Models sometimes wrap the answer in quotes or a code fence; unwrap. */
export function stripWrapping(text: string): string {
  let t = text.trim();
  t = t.replace(/^```[a-z]*\s*/i, '').replace(/\s*```$/, '').trim();
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith('”') && t.endsWith('”'))) t = t.slice(1, -1).trim();
  // a leading label such as "Korrigerad text:" is not part of the transcript
  t = t.replace(/^(korrigerad(?:e)? (?:text|transkription)|transkription|text)\s*:\s*/i, '');
  return t;
}

const STRUCTURE_PROMPT = [
  'Du strukturerar en transkriberad patientberättelse på svenska till journalfält.',
  'Använd ENDAST information som finns i texten. Hitta inte på. Lämna fält tomma (null eller []) när texten saknar uppgiften.',
  'Svara med ett JSON-objekt med exakt dessa nycklar:',
  '{"kontaktorsak": string|null, "anamnes": string|null, "aktuella_lakemedel": [{"namn": string, "dos": string|null, "frekvens": string|null}], "allergier": [string], "symtom": [string], "oklarheter": [string]}',
  '"oklarheter" listar sådant som är otydligt eller motsägelsefullt i texten. Inga kommentarer utanför JSON.',
].join('\n');
