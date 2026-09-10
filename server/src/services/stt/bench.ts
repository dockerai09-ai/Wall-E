// Benchmark harness: run a manifest of (audio, reference) cases through one
// pipeline variant, or score transcripts captured by hand for an arm that
// cannot be scripted (Claude Pro), and score both with the same metrics.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { transcribeClinical, variantKey, type SttVariant, type SttCache } from './pipeline.js';
import { aggregate, scoreCase, frequentMisses, type AggregateMetrics, type CaseScore } from './metrics.js';
import { loadGlossary, type Glossary } from './glossary.js';
import { insertRun, type SttRunRow } from './store.js';

export interface BenchCase {
  id: string;
  reference: string;
  terms?: string[];
  pii?: string[];
  tags?: string[];
  /** Absolute path to the audio file (from the synthesised manifest). */
  audio?: string;
}

export interface CaseResult {
  id: string;
  hypothesis: string | null;
  rawHypothesis: string | null;
  error: string | null;
  latencyMs: number | null;
  correctionRejected: boolean;
  rejectedText: string | null;
  cached: boolean;
  stt: { platform: string; model: string } | null;
  correction: { platform: string; model: string } | null;
  score: CaseScore | null;
  tags: string[];
}

export interface BenchRun {
  profile: string;
  arm: string;
  variant: SttVariant | null;
  glossaryHash: string | null;
  metrics: AggregateMetrics;
  cases: CaseResult[];
  startedAt: number;
  finishedAt: number;
  /** Reference words most often wrong, for the learning step and the report. */
  misses: { term: string; count: number }[];
}

export function parseManifest(text: string, baseDir: string): BenchCase[] {
  const out: BenchCase[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as BenchCase;
    if (!row.id || typeof row.reference !== 'string') throw new Error(`manifest row without id/reference: ${line.slice(0, 80)}`);
    if (row.audio && !path.isAbsolute(row.audio)) row.audio = path.resolve(baseDir, row.audio);
    out.push(row);
  }
  return out;
}

export function loadManifest(file: string): BenchCase[] {
  return parseManifest(fs.readFileSync(file, 'utf8'), path.dirname(file));
}

export function filterCases(cases: BenchCase[], opts: { limit?: number; tags?: string[]; ids?: string[] } = {}): BenchCase[] {
  let out = cases;
  if (opts.ids?.length) out = out.filter(c => opts.ids!.includes(c.id));
  if (opts.tags?.length) out = out.filter(c => (c.tags ?? []).some(t => opts.tags!.includes(t)));
  if (opts.limit && opts.limit > 0) out = out.slice(0, opts.limit);
  return out;
}

/** A JSON-file cache under the profile directory (git-ignored). */
export class FileCache implements SttCache {
  hits = 0;
  misses = 0;
  constructor(readonly dir: string) { fs.mkdirSync(dir, { recursive: true }); }
  private file(key: string): string {
    return path.join(this.dir, crypto.createHash('sha256').update(key).digest('hex').slice(0, 40) + '.json');
  }
  get(key: string): unknown | null {
    const f = this.file(key);
    if (!fs.existsSync(f)) { this.misses++; return null; }
    try { this.hits++; return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return null; }
  }
  set(key: string, value: unknown): void {
    fs.writeFileSync(this.file(key), JSON.stringify(value));
  }
}

export class MemoryCache implements SttCache {
  private m = new Map<string, unknown>();
  get(key: string) { return this.m.get(key) ?? null; }
  set(key: string, value: unknown) { this.m.set(key, value); }
}

/** Minimum spacing between real provider calls. */
export function makeThrottle(rpm: number): () => Promise<void> {
  const interval = rpm > 0 ? Math.ceil(60_000 / rpm) : 0;
  let last = 0;
  return async () => {
    if (!interval) return;
    const wait = last + interval - Date.now();
    if (wait > 0) await new Promise(r => setTimeout(r, wait));
    last = Date.now();
  };
}

export interface RunArmOptions {
  profile: string;
  arm?: string;
  glossary?: Glossary;
  cache?: SttCache;
  rpm?: number;
  waitForResetSeconds?: number;
  onProgress?: (done: number, total: number, r: CaseResult) => void;
}

export async function runArm(cases: BenchCase[], variant: SttVariant, opts: RunArmOptions): Promise<BenchRun> {
  const glossary = opts.glossary ?? loadGlossary(opts.profile);
  const throttle = makeThrottle(opts.rpm ?? 0);
  const startedAt = Date.now();
  const results: CaseResult[] = [];
  let rejections = 0;
  for (const c of cases) {
    const base: CaseResult = { id: c.id, hypothesis: null, rawHypothesis: null, error: null, latencyMs: null, correctionRejected: false, rejectedText: null, cached: false, stt: null, correction: null, score: null, tags: c.tags ?? [] };
    if (!c.audio || !fs.existsSync(c.audio)) {
      base.error = `audio not found: ${c.audio ?? '(none)'}`;
      results.push(base);
      opts.onProgress?.(results.length, cases.length, base);
      continue;
    }
    try {
      const out = await transcribeClinical(
        { audio: fs.readFileSync(c.audio), filename: path.basename(c.audio), mimeType: 'audio/wav' },
        variant,
        { profile: opts.profile, glossary, cache: opts.cache, throttle, waitForResetSeconds: opts.waitForResetSeconds ?? 120, persist: false, actor: 'bench' },
      );
      base.hypothesis = out.finalText;
      base.rawHypothesis = out.rawText;
      base.latencyMs = out.latencyMs;
      base.correctionRejected = out.correctionRejected;
      base.rejectedText = out.rejectedText;
      base.cached = out.sttCached;
      base.stt = { platform: out.stt.platform, model: out.stt.model };
      base.correction = out.correction ? { platform: out.correction.platform, model: out.correction.model } : null;
      base.score = scoreCase({ id: c.id, reference: c.reference, hypothesis: out.finalText, terms: c.terms, pii: c.pii });
      if (out.correctionRejected) rejections++;
    } catch (err: any) {
      base.error = String(err?.message ?? err).slice(0, 300);
    }
    results.push(base);
    opts.onProgress?.(results.length, cases.length, base);
  }
  return finishRun(opts.profile, opts.arm ?? 'wall-e', variant, glossary.hash, results, startedAt, rejections);
}

function finishRun(profile: string, arm: string, variant: SttVariant | null, glossaryHash: string | null, results: CaseResult[], startedAt: number, rejections: number): BenchRun {
  const scores = results.filter(r => r.score).map(r => r.score as CaseScore);
  const metrics = aggregate(scores, {
    errors: results.filter(r => r.error).length,
    latenciesMs: results.filter(r => r.latencyMs != null && !r.cached).map(r => r.latencyMs as number),
    correctionRejections: rejections,
    totalCases: results.length,
  });
  return { profile, arm, variant, glossaryHash, metrics, cases: results, startedAt, finishedAt: Date.now(), misses: frequentMisses(scores, 1, 60) };
}

export interface ManualHypothesis { id: string; hypothesis: string; source?: string; captured_at?: string }

export function parseArmFile(text: string): ManualHypothesis[] {
  const out: ManualHypothesis[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as ManualHypothesis;
    if (!row.id || typeof row.hypothesis !== 'string') throw new Error(`arm row without id/hypothesis: ${line.slice(0, 80)}`);
    if (/^<paste /.test(row.hypothesis.trim())) throw new Error(`arm row ${row.id} still holds the template placeholder`);
    out.push(row);
  }
  return out;
}

/** Score hand-captured transcripts (the Claude Pro arm) against the cases.
 *  Cases without a captured line are reported as errors, never skipped. */
export function scoreManualArm(cases: BenchCase[], hyps: ManualHypothesis[], profile: string, arm: string): BenchRun {
  const byId = new Map(hyps.map(h => [h.id, h]));
  const startedAt = Date.now();
  const results: CaseResult[] = cases.map(c => {
    const h = byId.get(c.id);
    const base: CaseResult = { id: c.id, hypothesis: null, rawHypothesis: null, error: null, latencyMs: null, correctionRejected: false, rejectedText: null, cached: false, stt: null, correction: null, score: null, tags: c.tags ?? [] };
    if (!h) { base.error = 'no transcript captured for this case'; return base; }
    base.hypothesis = h.hypothesis;
    base.rawHypothesis = h.hypothesis;
    base.score = scoreCase({ id: c.id, reference: c.reference, hypothesis: h.hypothesis, terms: c.terms, pii: c.pii });
    return base;
  });
  return finishRun(profile, arm, null, null, results, startedAt, 0);
}

export function persistRun(run: BenchRun, opts: { kind: 'bench' | 'loop'; loopId?: string | null; iteration?: number | null } = { kind: 'bench' }): SttRunRow {
  return insertRun({
    kind: opts.kind,
    profile: run.profile,
    arm: run.arm,
    loop_id: opts.loopId ?? null,
    iteration: opts.iteration ?? null,
    variant_json: JSON.stringify(run.variant ? { ...run.variant, key: variantKey(run.variant), glossaryHash: run.glossaryHash } : null),
    metrics_json: JSON.stringify(run.metrics),
    // per-case scores only: no reference or hypothesis text in the database
    cases_json: JSON.stringify(run.cases.map(c => ({ id: c.id, error: c.error, latencyMs: c.latencyMs, correctionRejected: c.correctionRejected, cached: c.cached, stt: c.stt, correction: c.correction, score: c.score ? { wer: c.score.wer, cer: c.score.cer, termRecall: c.score.termRecall, piiLeaked: c.score.piiLeaked, missed: c.score.missed } : null }))),
    status: run.metrics.scored > 0 ? 'done' : 'error',
  });
}

export function runFromRow(row: SttRunRow): { arm: string; metrics: AggregateMetrics; variant: SttVariant | null; createdAt: number; id: number } {
  const variant = JSON.parse(row.variant_json) as (SttVariant & { key?: string; glossaryHash?: string }) | null;
  if (variant) { delete variant.key; delete variant.glossaryHash; }
  return { id: row.id, arm: row.arm, metrics: JSON.parse(row.metrics_json) as AggregateMetrics, variant, createdAt: row.created_at_ms };
}

// ------------------------------------------------------------- report -----

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

export function metricsTable(rows: { label: string; metrics: AggregateMetrics }[]): string {
  const lines = [
    '| Arm | WER ↓ | CER ↓ | Term recall ↑ | PII leaks ↓ | Cases (err) | p50 / p95 latency | Correction rejected |',
    '|---|---|---|---|---|---|---|---|',
  ];
  for (const r of rows) {
    const m = r.metrics;
    lines.push(`| ${r.label} | ${pct(m.wer)} | ${pct(m.cer)} | ${pct(m.termRecall)} | ${pct(m.piiLeakRate)} | ${m.scored} (${m.errors}) | ${m.latencyP50Ms} / ${m.latencyP95Ms} ms | ${m.correctionRejections} |`);
  }
  return lines.join('\n');
}

export function benchReportMarkdown(runs: BenchRun[], opts: { title?: string; worst?: number } = {}): string {
  const out: string[] = [`# ${opts.title ?? 'Speech-to-text benchmark'}`, ''];
  out.push(metricsTable(runs.map(r => ({ label: r.variant ? `${r.arm} (${variantKey(r.variant)})` : r.arm, metrics: r.metrics }))));
  out.push('');
  out.push('WER/CER are corpus-level (total edits over total reference words/characters) after Swedish normalisation: number words and digits, dose units, and spelling variants such as igår/i går compare equal. Term recall counts glossary terms recovered verbatim. PII leaks count protected values (personnummer, phone) surviving into the final text.');
  for (const r of runs) {
    out.push('', `## ${r.arm}`, '');
    if (r.variant) out.push(`Variant: \`${variantKey(r.variant)}\`; glossary ${r.glossaryHash ?? '-'}.`, '');
    const worst = r.cases.filter(c => c.score).sort((a, b) => (b.score!.wer - a.score!.wer)).slice(0, opts.worst ?? 5);
    if (worst.length) {
      out.push('Worst cases:', '');
      out.push('| Case | WER | Missed words | Output |', '|---|---|---|---|');
      for (const c of worst) out.push(`| ${c.id} | ${pct(c.score!.wer)} | ${c.score!.missed.join(', ') || '-'} | ${(c.hypothesis ?? '').replace(/\|/g, '\\|').slice(0, 160)} |`);
    }
    const errors = r.cases.filter(c => c.error);
    if (errors.length) {
      out.push('', 'Errors:', '');
      for (const c of errors) out.push(`- ${c.id}: ${c.error}`);
    }
    if (r.misses.length) out.push('', `Most-missed reference words: ${r.misses.slice(0, 12).map(m => `${m.term} (${m.count})`).join(', ')}`);
  }
  return out.join('\n') + '\n';
}

export function compare(a: AggregateMetrics, b: AggregateMetrics): { wer: number; cer: number; termRecall: number; piiLeakRate: number } {
  return { wer: round(a.wer - b.wer), cer: round(a.cer - b.cer), termRecall: round(a.termRecall - b.termRecall), piiLeakRate: round(a.piiLeakRate - b.piiLeakRate) };
}

const round = (n: number) => Math.round(n * 10000) / 10000;
