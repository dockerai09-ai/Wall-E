// The Wall-E clinical LLM: a virtual model (`wall-e/<profile>`) served by
// /v1/chat/completions. It is not a fine-tuned network; it is a composition
// the router already knows how to run — the clinical STT pipeline for audio
// parts, the knowledge module for grounded questions, and a governed chat
// model wearing a persona for everything else — presented to clients as one
// model id so any OpenAI-compatible tool can use it.
//
//   user turn with audio  → document mode: transcript + journal-field draft
//   text question + KB    → question mode: cited answer or refusal (RAG)
//   anything else         → chat mode: persona reply through the router
//
// Output text is redacted with the governance policy before it leaves, and
// every audio part leaves a hash-only provenance row (stt_transcripts).

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { parse as parseYaml } from 'yaml';
import { z } from 'zod';
import type { ChatMessage, TokenUsage } from '@freellmapi/shared/types.js';
import { contentToString } from '../../lib/content.js';
import { getPolicy, redactText } from '../knowledge/governance.js';
import { knowledgeChat } from '../knowledge/llm.js';
import { resolveBase } from '../knowledge/store.js';
import { answerQuestion, type KnowledgeAnswer } from '../knowledge/rag.js';
import { KnowledgeError } from '../knowledge/config.js';
import { loadGlossary, profileDir } from './glossary.js';
import { parseVariant, transcribeClinical, normalizeSpokenNumbers, type SttVariant, type SttOutput, type StructuredNote } from './pipeline.js';

export const WALLE_MODEL_PREFIX = 'wall-e/';
const PROFILE_RE = /^[a-z0-9][a-z0-9_-]*$/i;

const DEFAULT_PERSONA_SV = [
  'Du är Wall-E, en klinisk dokumentationsassistent för svensk vård.',
  'Du hjälper vårdpersonal att transkribera, rätta och strukturera det patienter säger.',
  'Du ställer inga diagnoser och ger inga behandlingsråd. Du återger vad patienten sagt, markerar oklarheter och svarar på frågor bara utifrån det underlag du fått.',
  'Personnummer, telefonnummer och andra identifierare skrivs aldrig ut.',
  'Svara på svenska, kort och exakt.',
].join('\n');

const assistantSchema = z.object({
  name: z.string().min(1).optional(),
  description: z.string().default(''),
  persona: z.string().min(1).default(DEFAULT_PERSONA_SV),
  knowledge_base: z.string().min(1).nullable().default(null),
  stt: z.record(z.unknown()).default({}),
  output: z.object({
    transcript: z.boolean().default(true),
    structured: z.boolean().default(true),
  }).default({}),
  generation: z.object({
    model: z.string().min(1).nullable().default(null),
    temperature: z.number().min(0).max(2).default(0.2),
    max_tokens: z.number().int().positive().max(8192).default(1500),
    wait_for_reset_seconds: z.number().int().min(0).max(300).default(60),
  }).default({}),
});

export interface WallEAssistant {
  profile: string;
  /** The model id clients send: wall-e/<profile>. */
  id: string;
  name: string;
  description: string;
  language: string;
  persona: string;
  knowledgeBase: string | null;
  variant: SttVariant;
  output: { transcript: boolean; structured: boolean };
  generation: { model: string | null; temperature: number; maxTokens: number; waitForResetSeconds: number };
  /** Short hash of assistant.yaml + glossary, recorded in responses. */
  hash: string;
}

export function loadAssistant(profile: string): WallEAssistant {
  if (!PROFILE_RE.test(profile)) throw new KnowledgeError(`invalid profile '${profile}'`, 400);
  const dir = profileDir(profile);
  const glossary = loadGlossary(profile);
  const file = path.join(dir, 'assistant.yaml');
  const raw = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '';
  const parsed = assistantSchema.parse(raw.trim() ? (parseYaml(raw) ?? {}) : {});
  const variant = parseVariant({ sttModel: 'auto', structure: true, ...parsed.stt, language: (parsed.stt.language as string | undefined) ?? glossary.language });
  return {
    profile,
    id: `${WALLE_MODEL_PREFIX}${profile}`,
    name: parsed.name ?? `Wall-E ${profile}`,
    description: parsed.description,
    language: glossary.language,
    persona: parsed.persona.trim(),
    knowledgeBase: parsed.knowledge_base,
    variant,
    output: parsed.output,
    generation: { model: parsed.generation.model, temperature: parsed.generation.temperature, maxTokens: parsed.generation.max_tokens, waitForResetSeconds: parsed.generation.wait_for_reset_seconds },
    hash: crypto.createHash('sha256').update(raw).update(glossary.hash).digest('hex').slice(0, 12),
  };
}

/** Every profile directory with a glossary is a Wall-E model. */
export function listWallEModels(): WallEAssistant[] {
  const root = path.join(profileDir('x'), '..');
  if (!fs.existsSync(root)) return [];
  const out: WallEAssistant[] = [];
  for (const name of fs.readdirSync(root).sort()) {
    if (!PROFILE_RE.test(name) || !fs.existsSync(path.join(root, name, 'glossary.yaml'))) continue;
    try { out.push(loadAssistant(name)); } catch (err: any) { console.warn(`[wall-e] profile '${name}' skipped: ${err?.message ?? err}`); }
  }
  return out;
}

/** `wall-e/<profile>` → assistant, null for any other model id. */
export function resolveWallEModel(model: string | undefined | null): WallEAssistant | null {
  if (!model || !model.toLowerCase().startsWith(WALLE_MODEL_PREFIX)) return null;
  const profile = model.slice(WALLE_MODEL_PREFIX.length);
  if (!PROFILE_RE.test(profile)) throw new KnowledgeError(`Unknown model '${model}'. Wall-E models are wall-e/<profile>.`, 400);
  if (!fs.existsSync(path.join(profileDir(profile), 'glossary.yaml'))) {
    throw new KnowledgeError(`Unknown model '${model}'. Available: ${listWallEModels().map(a => a.id).join(', ') || '(none)'}.`, 400);
  }
  return loadAssistant(profile);
}

// ----------------------------------------------------------- audio parts --

export interface AudioPart {
  index: number;
  data: Buffer;
  format: string;
  mimeType: string;
  filename: string;
}

const AUDIO_MIME: Record<string, string> = { wav: 'audio/wav', mp3: 'audio/mpeg', m4a: 'audio/mp4', mp4: 'audio/mp4', flac: 'audio/flac', ogg: 'audio/ogg', oga: 'audio/ogg', webm: 'audio/webm', opus: 'audio/ogg', aac: 'audio/aac' };

function formatFromMime(mime: string): string {
  const m = mime.toLowerCase();
  for (const [fmt, mt] of Object.entries(AUDIO_MIME)) if (mt === m) return fmt;
  const sub = m.split('/')[1] ?? 'wav';
  return sub.replace(/^x-/, '');
}

/** Audio blocks in a message: OpenAI `input_audio` ({data, format}), or an
 *  `audio_url`/`audio` block carrying a data: URL. Remote URLs are refused —
 *  the server never fetches audio from the network on a client's behalf. */
export function extractAudioParts(content: unknown): AudioPart[] {
  if (!Array.isArray(content)) return [];
  const out: AudioPart[] = [];
  content.forEach((block, i) => {
    if (!block || typeof block !== 'object') return;
    const b = block as Record<string, unknown>;
    const type = String(b.type ?? '');
    if (type === 'input_audio' && b.input_audio && typeof b.input_audio === 'object') {
      const ia = b.input_audio as { data?: unknown; format?: unknown };
      if (typeof ia.data !== 'string' || !ia.data) throw new KnowledgeError(`input_audio block ${i} has no base64 data`, 400);
      const format = String(ia.format ?? 'wav').toLowerCase();
      out.push({ index: i, data: Buffer.from(ia.data, 'base64'), format, mimeType: AUDIO_MIME[format] ?? `audio/${format}`, filename: `part-${i}.${format}` });
      return;
    }
    if (type === 'audio_url' || type === 'audio') {
      const urlHolder = (b.audio_url ?? b.audio ?? b) as { url?: unknown; data?: unknown };
      const url = typeof urlHolder === 'object' && urlHolder && typeof urlHolder.url === 'string' ? urlHolder.url : typeof b.url === 'string' ? b.url : null;
      if (!url) return;
      const m = /^data:(audio\/[a-z0-9.+-]+);base64,(.+)$/i.exec(url);
      if (!m) throw new KnowledgeError(`audio block ${i} must be a data: URL with base64 audio; remote URLs are not fetched`, 400);
      const format = formatFromMime(m[1]);
      out.push({ index: i, data: Buffer.from(m[2], 'base64'), format, mimeType: m[1], filename: `part-${i}.${format}` });
    }
  });
  return out;
}

/** Text of a message with audio blocks replaced by a marker. */
function textWithoutAudio(content: unknown): string {
  if (!Array.isArray(content)) return contentToString(content);
  return content
    .map(block => {
      if (typeof block === 'string') return block;
      const type = String((block as Record<string, unknown>)?.type ?? '');
      if (type === 'input_audio' || type === 'audio_url' || type === 'audio') return '';
      return contentToString([block]);
    })
    .filter(Boolean)
    .join('\n');
}

// -------------------------------------------------------------- running ---

export interface AssistantInput {
  messages: ChatMessage[];
  actor: string;
  /** A client profile's enforced prompt; prepended to the persona. */
  enforcedSystemPrompt?: string | null;
  temperature?: number;
  maxTokens?: number;
}

export interface AssistantTranscript {
  part: number;
  text: string;
  structured: StructuredNote | null;
  structure_error: string | null;
  redactions: SttOutput['redactions'];
  correction: { model: string; applied: boolean } | null;
  provenance_id: number | null;
  stt: { model: string; duration_s: number | null; latency_ms: number };
}

export interface AssistantResult {
  mode: 'document' | 'question' | 'chat';
  content: string;
  transcripts: AssistantTranscript[];
  answer: { id: string; status: string; citations: KnowledgeAnswer['citations']; knowledge_base: string } | null;
  model: string | null;
  usage: TokenUsage;
  redactions: SttOutput['redactions'];
  latencyMs: number;
}

const ZERO: TokenUsage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
const addUsage = (a: TokenUsage, b: TokenUsage | null | undefined): TokenUsage => ({
  prompt_tokens: a.prompt_tokens + (b?.prompt_tokens ?? 0),
  completion_tokens: a.completion_tokens + (b?.completion_tokens ?? 0),
  total_tokens: a.total_tokens + (b?.total_tokens ?? 0),
});

function lastUserIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'user') return i;
  return -1;
}

export function renderDocument(t: AssistantTranscript, output: WallEAssistant['output']): string {
  const out: string[] = [];
  if (output.transcript) out.push('## Transkription', '', t.text || '_(tomt)_');
  if (output.structured) {
    out.push('', '## Journalanteckning (utkast)', '');
    const s = t.structured;
    if (!s) {
      out.push(`_Strukturering misslyckades: ${t.structure_error ?? 'okänt fel'}_`);
    } else {
      const meds = s.aktuella_lakemedel.map(m => [m.namn, m.dos, m.frekvens].filter(Boolean).join(' ')).join('; ');
      out.push(`**Kontaktorsak:** ${s.kontaktorsak ?? '–'}`);
      out.push(`**Anamnes:** ${s.anamnes ?? '–'}`);
      out.push(`**Aktuella läkemedel:** ${meds || '–'}`);
      out.push(`**Allergier:** ${s.allergier.join(', ') || '–'}`);
      out.push(`**Symtom:** ${s.symtom.join(', ') || '–'}`);
      out.push(`**Oklarheter / att följa upp:** ${s.oklarheter.join('; ') || '–'}`);
    }
  }
  if (t.redactions.length) out.push('', `_Maskerat: ${t.redactions.map(r => `${r.detector} ×${r.count}`).join(', ')}_`);
  return out.join('\n');
}

function systemPrompt(a: WallEAssistant, enforced: string | null | undefined): string {
  return [enforced?.trim(), a.persona].filter(Boolean).join('\n\n');
}

export async function runWallEAssistant(a: WallEAssistant, input: AssistantInput): Promise<AssistantResult> {
  const started = Date.now();
  const policy = getPolicy();
  const idx = lastUserIndex(input.messages);
  if (idx === -1) throw new KnowledgeError('a user message is required', 400);
  const last = input.messages[idx];
  const audio = extractAudioParts(last.content);
  const userText = textWithoutAudio(last.content).trim();
  let usage = ZERO;
  let model: string | null = null;
  const transcripts: AssistantTranscript[] = [];
  const redactions = new Map<string, number>();
  const noteRedactions = (rs: SttOutput['redactions']) => { for (const r of rs) redactions.set(r.detector, (redactions.get(r.detector) ?? 0) + r.count); };

  const finish = (mode: AssistantResult['mode'], content: string, answer: AssistantResult['answer'] = null): AssistantResult => {
    // the model's own words are redacted too: a persona reply must never
    // reintroduce an identifier the transcript step masked
    let text = content;
    if (a.variant.redact) {
      const r = redactText(normalizeSpokenNumbers(text), { ...policy, ingest: { ...policy.ingest, redact_pii: true } });
      text = r.text;
      noteRedactions(r.redactions);
    }
    return {
      mode, content: text, transcripts, answer, model, usage,
      redactions: [...redactions.entries()].map(([detector, count]) => ({ detector, count })),
      latencyMs: Date.now() - started,
    };
  };

  // ---- document mode: audio in the last user turn
  if (audio.length > 0) {
    for (const part of audio) {
      const out = await transcribeClinical(
        { audio: part.data, filename: part.filename, mimeType: part.mimeType },
        { ...a.variant, structure: a.output.structured || a.variant.structure },
        { profile: a.profile, actor: input.actor, persist: true, waitForResetSeconds: a.generation.waitForResetSeconds },
      );
      if (out.correction) { usage = addUsage(usage, out.correction.usage); model = `${out.correction.platform}/${out.correction.model}`; }
      noteRedactions(out.redactions);
      transcripts.push({
        part: part.index, text: out.finalText, structured: out.structured, structure_error: out.structureError, redactions: out.redactions,
        correction: out.correction ? { model: `${out.correction.platform}/${out.correction.model}`, applied: !out.correctionRejected } : null,
        provenance_id: out.provenanceId,
        stt: { model: `${out.stt.platform}/${out.stt.model}`, duration_s: out.stt.durationS, latency_ms: out.stt.latencyMs },
      });
    }
    const documents = transcripts.map(t => (transcripts.length > 1 ? `# Ljuddel ${t.part + 1}\n\n` : '') + renderDocument(t, a.output)).join('\n\n');
    if (!userText) return finish('document', documents);
    // the clinician also asked for something: let the persona work on the
    // transcript, with the deterministic document as the ground it stands on
    const r = await knowledgeChat([
      { role: 'system', content: systemPrompt(a, input.enforcedSystemPrompt) + '\n\nNedan följer transkriptionen (redan maskerad) och ett utkast till journalfält. Utför användarens instruktion utifrån dem. Hitta inte på uppgifter som saknas.' },
      { role: 'user', content: `${documents}\n\n---\nInstruktion: ${userText}` },
    ], { temperature: input.temperature ?? a.generation.temperature, maxTokens: input.maxTokens ?? a.generation.maxTokens, model: a.generation.model, timeoutMs: 90_000, waitForResetSeconds: a.generation.waitForResetSeconds });
    usage = addUsage(usage, r.usage);
    model = `${r.platform}/${r.modelId}`;
    return finish('document', `${documents}\n\n## Svar\n\n${r.text.trim()}`);
  }

  if (!userText) throw new KnowledgeError('the last user message has neither text nor audio', 400);

  // ---- question mode: grounded in the profile's knowledge base
  if (a.knowledgeBase) {
    const base = resolveBase(a.knowledgeBase);
    if (!base) throw new KnowledgeError(`knowledge base '${a.knowledgeBase}' configured for ${a.id} does not exist`, 500);
    const ans = await answerQuestion(base, userText, {
      actor: input.actor, kind: 'query', model: a.generation.model,
      systemPrompt: systemPrompt(a, input.enforcedSystemPrompt), temperature: input.temperature ?? a.generation.temperature,
    });
    usage = addUsage(usage, ans.usage);
    model = ans.model ? `${ans.model.platform}/${ans.model.modelId}` : null;
    const sources = ans.citations.length ? '\n\nKällor:\n' + ans.citations.map(c => `[${c.n}] ${c.title}${c.source ? ` — ${c.source}` : ''}`).join('\n') : '';
    return finish('question', ans.answer + sources, { id: ans.id, status: ans.status, citations: ans.citations, knowledge_base: base.slug });
  }

  // ---- chat mode: persona over the conversation (history flattened to text)
  const history: ChatMessage[] = input.messages
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({ role: m.role, content: m.role === 'user' ? (textWithoutAudio(m.content) || '[ljud]') : contentToString(m.content) }));
  const r = await knowledgeChat([
    { role: 'system', content: systemPrompt(a, input.enforcedSystemPrompt) },
    ...history.slice(-20),
  ], { temperature: input.temperature ?? a.generation.temperature, maxTokens: input.maxTokens ?? a.generation.maxTokens, model: a.generation.model, timeoutMs: 90_000, waitForResetSeconds: a.generation.waitForResetSeconds });
  usage = addUsage(usage, r.usage);
  model = `${r.platform}/${r.modelId}`;
  return finish('chat', r.text.trim());
}
