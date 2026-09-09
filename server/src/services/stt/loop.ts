// Loop engineering: an eval-driven optimisation loop over the clinical STT
// pipeline. One iteration = a coordinate-descent pass over the search space
// (each dimension tried one at a time from the incumbent), then a learning
// step that feeds the words the incumbent still misses back into the
// glossary, then a re-evaluation. Every candidate is scored on the same
// benchmark with the same metrics as the manual Claude Pro arm; the loop
// stops when it beats the target by the configured margin, when `patience`
// passes bring no gain, or at max_iterations. State and a report are
// written after every pass so an interrupted loop is inspectable.

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { parse as parseYaml } from 'yaml';
import { z } from 'zod';
import { parseVariant, variantKey, type SttVariant } from './pipeline.js';
import { runArm, persistRun, metricsTable, type BenchCase, type BenchRun, type RunArmOptions } from './bench.js';
import { loadGlossary, profileDir, saveLearnedTerms, allTerms, type Glossary } from './glossary.js';
import { frequentMisses, type AggregateMetrics, type CaseScore } from './metrics.js';
import { latestArmRun } from './store.js';

const loopSchema = z.object({
  profile: z.string().min(1),
  objective: z.enum(['wer', 'cer', 'term_error']).default('wer'),
  guardrails: z.object({
    min_term_recall: z.number().min(0).max(1).default(0.85),
    max_pii_leak_rate: z.number().min(0).max(1).default(0),
    max_p95_latency_ms: z.number().positive().default(30_000),
    max_correction_rejections: z.number().min(0).max(1).default(0.2),
    // WER is computed over scored cases only, so a candidate that errors on
    // half the benchmark could otherwise win on the half it managed
    max_error_rate: z.number().min(0).max(1).default(0.1),
  }).default({}),
  target: z.object({
    arm: z.string().nullable().default('claude-pro'),
    wer: z.number().min(0).max(1).default(0.08),
    margin: z.number().min(0).max(1).default(0.01),
  }).default({}),
  budget: z.object({
    max_iterations: z.number().int().min(1).max(50).default(4),
    patience: z.number().int().min(1).default(2),
    rpm: z.number().min(0).default(18),
  }).default({}),
  search: z.object({
    stt_model: z.array(z.string()).default(['auto']),
    prompt: z.array(z.enum(['none', 'glossary'])).default(['glossary', 'none']),
    correction: z.array(z.enum(['off', 'llm'])).default(['llm', 'off']),
    correction_model: z.array(z.string().nullable()).default([null]),
    temperature: z.array(z.number().min(0).max(1)).default([0]),
  }).default({}),
  start: z.record(z.unknown()).default({}),
  learn: z.object({
    enabled: z.boolean().default(true),
    min_occurrences: z.number().int().min(1).default(2),
    max_terms: z.number().int().min(1).default(40),
  }).default({}),
});

export type LoopConfig = z.infer<typeof loopSchema>;

export function loadLoopConfig(profile: string, file?: string): LoopConfig {
  const f = file ?? path.join(profileDir(profile), 'loop.yaml');
  const raw = fs.existsSync(f) ? parseYaml(fs.readFileSync(f, 'utf8')) : {};
  return loopSchema.parse({ profile, ...(raw ?? {}) });
}

const DIMENSIONS: { key: keyof LoopConfig['search']; field: keyof SttVariant }[] = [
  { key: 'stt_model', field: 'sttModel' },
  { key: 'prompt', field: 'prompt' },
  { key: 'correction', field: 'correction' },
  { key: 'correction_model', field: 'correctionModel' },
  { key: 'temperature', field: 'temperature' },
];

export function objectiveOf(m: AggregateMetrics, objective: LoopConfig['objective']): number {
  if (objective === 'cer') return m.cer;
  if (objective === 'term_error') return 1 - m.termRecall;
  return m.wer;
}

export function guardrailFailures(m: AggregateMetrics, g: LoopConfig['guardrails']): string[] {
  const out: string[] = [];
  if (m.scored === 0) out.push('no case was scored');
  if (m.cases > 0 && m.errors / m.cases > g.max_error_rate) out.push(`error rate ${m.errors}/${m.cases} > ${g.max_error_rate}`);
  if (m.termRecall < g.min_term_recall) out.push(`term recall ${m.termRecall} < ${g.min_term_recall}`);
  if (m.piiLeakRate > g.max_pii_leak_rate) out.push(`PII leak rate ${m.piiLeakRate} > ${g.max_pii_leak_rate}`);
  if (m.latencyP95Ms > g.max_p95_latency_ms) out.push(`p95 latency ${m.latencyP95Ms}ms > ${g.max_p95_latency_ms}ms`);
  if (m.scored > 0 && m.correctionRejections / m.scored > g.max_correction_rejections) out.push(`correction rejections ${m.correctionRejections}/${m.scored} > ${g.max_correction_rejections}`);
  return out;
}

export interface LoopTarget {
  arm: string | null;
  /** Objective value to beat. */
  value: number;
  source: 'arm' | 'config';
  runId: number | null;
}

export function resolveTarget(config: LoopConfig): LoopTarget {
  if (config.target.arm) {
    const row = latestArmRun(config.profile, config.target.arm);
    if (row) {
      const m = JSON.parse(row.metrics_json) as AggregateMetrics;
      return { arm: config.target.arm, value: objectiveOf(m, config.objective), source: 'arm', runId: row.id };
    }
  }
  return { arm: config.target.arm, value: config.target.wer, source: 'config', runId: null };
}

export interface Evaluation {
  variant: SttVariant;
  key: string;
  metrics: AggregateMetrics;
  objective: number;
  guardrails: string[];
  runId: number | null;
  glossaryHash: string | null;
  iteration: number;
  role: 'start' | 'candidate' | 'relearn';
  dimension?: string;
}

export interface LoopState {
  loopId: string;
  profile: string;
  objective: LoopConfig['objective'];
  target: LoopTarget;
  startedAt: number;
  updatedAt: number;
  status: 'running' | 'target_met' | 'converged' | 'budget_exhausted' | 'error';
  iteration: number;
  evaluations: Evaluation[];
  best: Evaluation | null;
  learned: { term: string; count: number; iteration: number }[];
  notes: string[];
}

export interface LoopOptions {
  cases: BenchCase[];
  /** Evaluate a variant; defaults to runArm with the loop's rpm and cache. */
  evaluate?: (variant: SttVariant, glossary: Glossary) => Promise<BenchRun>;
  cache?: RunArmOptions['cache'];
  outDir?: string;
  log?: (line: string) => void;
  maxIterations?: number;
  /** Skip DB persistence (tests). */
  persist?: boolean;
}

export async function runLoop(config: LoopConfig, opts: LoopOptions): Promise<LoopState> {
  const log = opts.log ?? (() => {});
  const loopId = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15) + '-' + crypto.randomBytes(2).toString('hex');
  const outDir = opts.outDir ?? path.join(profileDir(config.profile), 'runs', `loop-${loopId}`);
  fs.mkdirSync(outDir, { recursive: true });
  const target = resolveTarget(config);
  const maxIterations = opts.maxIterations ?? config.budget.max_iterations;
  const state: LoopState = {
    loopId, profile: config.profile, objective: config.objective, target, startedAt: Date.now(), updatedAt: Date.now(),
    status: 'running', iteration: 0, evaluations: [], best: null, learned: [], notes: [],
  };
  const save = () => {
    state.updatedAt = Date.now();
    fs.writeFileSync(path.join(outDir, 'state.json'), JSON.stringify(state, null, 2));
    fs.writeFileSync(path.join(outDir, 'report.md'), loopReportMarkdown(state, config));
  };
  const seen = new Map<string, Evaluation>();
  let glossary = loadGlossary(config.profile);

  const evaluate = async (variant: SttVariant, role: Evaluation['role'], dimension?: string): Promise<Evaluation> => {
    const key = `${variantKey(variant)}@${variant.prompt === 'glossary' || variant.correction === 'llm' ? glossary.hash : '-'}`;
    const prior = seen.get(key);
    if (prior) return { ...prior, role, dimension, iteration: state.iteration };
    log(`  evaluating ${variantKey(variant)} (${role}${dimension ? ':' + dimension : ''})`);
    const run = opts.evaluate
      ? await opts.evaluate(variant, glossary)
      : await runArm(opts.cases, variant, { profile: config.profile, arm: 'wall-e', glossary, cache: opts.cache, rpm: config.budget.rpm, onProgress: (d, t, r) => { if (r.error) log(`    ${r.id}: ${r.error}`); else if (d % 10 === 0 || d === t) log(`    ${d}/${t}`); } });
    const row = opts.persist === false ? null : persistRun(run, { kind: 'loop', loopId, iteration: state.iteration });
    const ev: Evaluation = {
      variant, key, metrics: run.metrics, objective: objectiveOf(run.metrics, config.objective),
      guardrails: guardrailFailures(run.metrics, config.guardrails), runId: row?.id ?? null, glossaryHash: run.glossaryHash,
      iteration: state.iteration, role, dimension,
    };
    seen.set(key, ev);
    state.evaluations.push(ev);
    lastRuns.set(key, run);
    log(`    → ${config.objective}=${ev.objective.toFixed(4)} termRecall=${run.metrics.termRecall} pii=${run.metrics.piiLeakRate} errors=${run.metrics.errors}${ev.guardrails.length ? ' GUARDRAIL: ' + ev.guardrails.join('; ') : ''}`);
    save();
    return ev;
  };
  const lastRuns = new Map<string, BenchRun>();
  const better = (a: Evaluation, b: Evaluation | null) => a.guardrails.length === 0 && (b == null || a.objective < b.objective - 1e-9);
  const targetMet = (e: Evaluation | null) => e != null && e.guardrails.length === 0 && e.objective <= target.value - config.target.margin;

  try {
    log(`loop ${loopId}: objective=${config.objective}, target ${target.value.toFixed(4)} from ${target.source === 'arm' ? `arm '${target.arm}' (run #${target.runId})` : 'loop.yaml'}, margin ${config.target.margin}`);
    let incumbent = parseVariant(config.start);
    const best = await evaluate(incumbent, 'start');
    if (best.guardrails.length) state.notes.push(`start variant fails guardrails: ${best.guardrails.join('; ')}`);
    state.best = best.guardrails.length ? null : best;
    let stale = 0;
    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      state.iteration = iteration;
      log(`iteration ${iteration}/${maxIterations}`);
      let improved = false;
      for (const dim of DIMENSIONS) {
        for (const value of config.search[dim.key] as unknown[]) {
          if (value === incumbent[dim.field]) continue;
          const candidate = { ...incumbent, [dim.field]: value } as SttVariant;
          const ev = await evaluate(candidate, 'candidate', dim.key);
          if (better(ev, state.best)) {
            state.best = ev;
            incumbent = candidate;
            improved = true;
            log(`  ✓ ${dim.key}=${String(value)} adopted (${config.objective}=${ev.objective.toFixed(4)})`);
          }
        }
      }
      // learning step: feed the incumbent's misses back into the glossary
      if (config.learn.enabled && state.best) {
        const run = lastRuns.get(state.best.key);
        const scores = (run?.cases ?? []).filter(c => c.score).map(c => c.score as CaseScore);
        const known = new Set(allTerms(glossary).map(t => t.toLowerCase()));
        const fresh = frequentMisses(scores, config.learn.min_occurrences, config.learn.max_terms).filter(m => !known.has(m.term.toLowerCase()));
        if (fresh.length) {
          const before = fs.existsSync(path.join(glossary.dir, 'glossary.learned.yaml')) ? fs.readFileSync(path.join(glossary.dir, 'glossary.learned.yaml'), 'utf8') : null;
          glossary = saveLearnedTerms(config.profile, fresh);
          log(`  learned ${fresh.length} term(s): ${fresh.map(f => f.term).join(', ')}`);
          if (incumbent.prompt === 'glossary' || incumbent.correction === 'llm') {
            const ev = await evaluate(incumbent, 'relearn');
            if (better(ev, state.best)) {
              state.best = ev;
              improved = true;
              for (const f of fresh) state.learned.push({ ...f, iteration });
              log(`  ✓ learned terms kept (${config.objective}=${ev.objective.toFixed(4)})`);
            } else {
              // roll back: the extra prompt words did not help
              const file = path.join(glossary.dir, 'glossary.learned.yaml');
              if (before == null) fs.rmSync(file, { force: true }); else fs.writeFileSync(file, before);
              glossary = loadGlossary(config.profile);
              state.notes.push(`iteration ${iteration}: learned terms rolled back (no improvement)`);
              log('  ✗ learned terms rolled back (no improvement)');
            }
          } else {
            for (const f of fresh) state.learned.push({ ...f, iteration });
          }
        }
      }
      save();
      if (targetMet(state.best)) { state.status = 'target_met'; break; }
      if (!improved) {
        stale++;
        if (stale >= config.budget.patience) { state.status = 'converged'; break; }
      } else {
        stale = 0;
      }
    }
    if (state.status === 'running') state.status = 'budget_exhausted';
  } catch (err: any) {
    state.status = 'error';
    state.notes.push(`error: ${err?.message ?? err}`);
    throw err;
  } finally {
    save();
    log(`loop ${loopId} ${state.status}; best ${state.best ? `${config.objective}=${state.best.objective.toFixed(4)} (${variantKey(state.best.variant)})` : 'none'}; state in ${outDir}`);
  }
  return state;
}

export function loopReportMarkdown(state: LoopState, config: LoopConfig): string {
  const out: string[] = [`# STT loop ${state.loopId} — ${state.profile}`, ''];
  out.push(`Status: **${state.status}** after ${state.iteration} iteration(s), ${state.evaluations.length} evaluation(s).`);
  out.push(`Objective: ${state.objective}. Target: ${state.target.value.toFixed(4)} (${state.target.source === 'arm' ? `arm '${state.target.arm}', run #${state.target.runId}` : 'from loop.yaml'}), margin ${config.target.margin}.`);
  if (state.best) {
    out.push('', `Best: \`${variantKey(state.best.variant)}\` — ${state.objective} ${state.best.objective.toFixed(4)}${state.best.runId ? ` (run #${state.best.runId})` : ''}.`);
    out.push('', metricsTable([{ label: 'best', metrics: state.best.metrics }]));
  } else {
    out.push('', 'No variant passed the guardrails yet.');
  }
  out.push('', '## Evaluations', '', '| # | Iter | Role | Variant | ' + state.objective + ' | Term recall | PII leaks | Errors | Guardrails |', '|---|---|---|---|---|---|---|---|---|');
  state.evaluations.forEach((e, i) => {
    out.push(`| ${i + 1} | ${e.iteration} | ${e.role}${e.dimension ? ':' + e.dimension : ''} | \`${variantKey(e.variant)}\` | ${e.objective.toFixed(4)} | ${e.metrics.termRecall} | ${e.metrics.piiLeakRate} | ${e.metrics.errors} | ${e.guardrails.join('; ') || 'ok'} |`);
  });
  if (state.learned.length) out.push('', '## Learned terms', '', state.learned.map(l => `- ${l.term} (missed in ${l.count} cases, iteration ${l.iteration})`).join('\n'));
  if (state.notes.length) out.push('', '## Notes', '', state.notes.map(n => `- ${n}`).join('\n'));
  return out.join('\n') + '\n';
}
