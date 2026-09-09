// Domain vocabulary for a speech-to-text profile. A profile is a directory
// under knowledge/stt/<name>/ holding glossary.yaml (curated) and, once the
// loop has run, glossary.learned.yaml (terms the benchmark showed the
// recogniser missing). Both feed the Whisper prompt and the correction prompt.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { parse as parseYaml, stringify as stringifyYaml } from 'yaml';
import { z } from 'zod';
import { REPO_ROOT } from '../knowledge/config.js';

const termSchema = z.object({
  term: z.string().min(1),
  aliases: z.array(z.string()).default([]),
});

const glossarySchema = z.object({
  name: z.string().min(1),
  version: z.number().int().nonnegative().default(1),
  language: z.string().min(2).default('sv'),
  context: z.string().default(''),
  categories: z.record(z.array(termSchema)).default({}),
});

const learnedSchema = z.object({
  terms: z.array(z.object({ term: z.string().min(1), count: z.number().int().nonnegative().default(0), learned_at: z.string().optional() })).default([]),
});

export type GlossaryTerm = z.infer<typeof termSchema>;

export interface Glossary {
  name: string;
  version: number;
  language: string;
  context: string;
  categories: Record<string, GlossaryTerm[]>;
  learned: { term: string; count: number; learned_at?: string }[];
  /** Short hash of the effective content, recorded in provenance. */
  hash: string;
  dir: string;
}

export function profileDir(profile: string): string {
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(profile)) throw new Error(`invalid profile name '${profile}'`);
  return path.join(REPO_ROOT, 'knowledge', 'stt', profile);
}

export function loadGlossary(profile: string): Glossary {
  const dir = profileDir(profile);
  const file = path.join(dir, 'glossary.yaml');
  if (!fs.existsSync(file)) throw new Error(`glossary not found: ${file}`);
  const raw = fs.readFileSync(file, 'utf8');
  const parsed = glossarySchema.parse(parseYaml(raw));
  const learnedFile = path.join(dir, 'glossary.learned.yaml');
  const learnedRaw = fs.existsSync(learnedFile) ? fs.readFileSync(learnedFile, 'utf8') : '';
  const learned = learnedRaw.trim() ? learnedSchema.parse(parseYaml(learnedRaw) ?? {}).terms : [];
  const hash = crypto.createHash('sha256').update(raw).update('\n--learned--\n').update(learnedRaw).digest('hex').slice(0, 12);
  return { ...parsed, learned, hash, dir };
}

export function allTerms(g: Glossary): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const terms of Object.values(g.categories)) {
    for (const t of terms) {
      const k = t.term.toLowerCase();
      if (!seen.has(k)) { seen.add(k); out.push(t.term); }
    }
  }
  for (const t of g.learned) {
    const k = t.term.toLowerCase();
    if (!seen.has(k)) { seen.add(k); out.push(t.term); }
  }
  return out;
}

/** Whisper reads the prompt as preceding speech, so it must look like
 *  transcript text: a short context sentence followed by the vocabulary as a
 *  comma list. Whisper keeps only the last ~224 tokens; learned terms go last
 *  so they are never the ones truncated. */
export function whisperPrompt(g: Glossary, maxChars = 700): string {
  const curated = allTerms({ ...g, learned: [] });
  const learned = g.learned.map(t => t.term).filter(t => !curated.some(c => c.toLowerCase() === t.toLowerCase()));
  const head = g.context.trim();
  let body = '';
  const budget = maxChars - head.length - 2;
  const learnedPart = learned.join(', ');
  const remaining = budget - (learnedPart ? learnedPart.length + 2 : 0);
  const parts: string[] = [];
  let used = 0;
  for (const t of curated) {
    if (used + t.length + 2 > remaining) break;
    parts.push(t);
    used += t.length + 2;
  }
  body = [parts.join(', '), learnedPart].filter(Boolean).join(', ');
  return [head, body ? body + '.' : ''].filter(Boolean).join(' ');
}

/** System prompt for the LLM correction pass. */
export function correctionSystemPrompt(g: Glossary): string {
  const lines: string[] = [];
  for (const [cat, terms] of Object.entries(g.categories)) {
    const items = terms.map(t => (t.aliases.length ? `${t.term} (hörs ofta som: ${t.aliases.join(', ')})` : t.term));
    lines.push(`${cat}: ${items.join('; ')}`);
  }
  if (g.learned.length) lines.push(`inlärda: ${g.learned.map(t => t.term).join('; ')}`);
  return [
    'Du korrigerar en automatisk transkription av ett patientsamtal på svenska.',
    'Rätta ENDAST igenkänningsfel: felhörda läkemedelsnamn, medicinska termer, siffror, doser och enheter.',
    'Ändra inte innehåll, ordföljd eller patientens formuleringar. Lägg inte till något. Ta inte bort något.',
    'Sammanfatta inte. Svara inte på patienten. Skriv inga kommentarer.',
    'Skriv doser med siffror och förkortade enheter (t.ex. "50 mg").',
    'Sammansatta ord som delats fel skrivs ihop igen ("person nummer" → "personnummer", "vård centralen" → "vårdcentralen").',
    'Om transkriptionen redan är korrekt, returnera den oförändrad.',
    'Svara med enbart den korrigerade texten.',
    '',
    'Ordlista:',
    ...lines,
  ].join('\n');
}

/** Merge learned terms into glossary.learned.yaml (idempotent, keeps counts). */
export function saveLearnedTerms(profile: string, terms: { term: string; count: number }[]): Glossary {
  const dir = profileDir(profile);
  const file = path.join(dir, 'glossary.learned.yaml');
  const existing = fs.existsSync(file) && fs.readFileSync(file, 'utf8').trim()
    ? learnedSchema.parse(parseYaml(fs.readFileSync(file, 'utf8')) ?? {}).terms
    : [];
  const byTerm = new Map(existing.map(t => [t.term.toLowerCase(), t]));
  const today = new Date().toISOString().slice(0, 10);
  for (const t of terms) {
    const k = t.term.toLowerCase();
    const prev = byTerm.get(k);
    if (prev) prev.count = Math.max(prev.count, t.count);
    else byTerm.set(k, { term: t.term, count: t.count, learned_at: today });
  }
  const out = { terms: [...byTerm.values()].sort((a, b) => b.count - a.count || a.term.localeCompare(b.term)) };
  fs.writeFileSync(file, '# Terms the benchmark loop found the recogniser missing. Generated; edit freely.\n' + stringifyYaml(out));
  return loadGlossary(profile);
}

export function glossaryStats(g: Glossary): { curated: number; learned: number; categories: string[] } {
  return {
    curated: allTerms({ ...g, learned: [] }).length,
    learned: g.learned.length,
    categories: Object.keys(g.categories),
  };
}
