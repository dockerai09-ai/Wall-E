// Evaluation: run a JSONL dataset of questions through the full pipeline and
// score retrieval (hit rate, recall@k, MRR), citations (precision against the
// expected sources), answers (an LLM judge for correctness and faithfulness,
// optional so retrieval-only runs are free), plus refusal/error rates and
// latency. Results are stored per run so quality can be tracked over time.

import crypto from 'crypto';
import type { ChatMessage } from '@freellmapi/shared/types.js';
import { z } from 'zod';
import { KnowledgeError } from './config.js';
import { audit, getPolicy } from './governance.js';
import { extractJsonObject, knowledgeChat } from './llm.js';
import { answerQuestion, type KnowledgeAnswer } from './rag.js';
import { finishEvalRun, getEvalRun, graphStats, insertEvalRun, type BaseRow, type EvalRunRow } from './store.js';

export const evalCaseSchema = z.object({
  id: z.string().min(1).optional(),
  question: z.string().min(1),
  expected_answer: z.string().optional(),
  expected_sources: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
});
export type EvalCase = z.infer<typeof evalCaseSchema> & { id: string };

export function parseJsonl(text: string): EvalCase[] {
  const out: EvalCase[] = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    let raw: unknown;
    try { raw = JSON.parse(trimmed); } catch { throw new KnowledgeError(`dataset line ${i + 1} is not valid JSON`); }
    const parsed = evalCaseSchema.safeParse(raw);
    if (!parsed.success) throw new KnowledgeError(`dataset line ${i + 1}: ${parsed.error.issues[0]?.message ?? 'invalid case'}`);
    out.push({ ...parsed.data, id: parsed.data.id ?? `case-${i + 1}` });
  });
  if (out.length === 0) throw new KnowledgeError('dataset has no cases');
  return out;
}

export interface EvalOptions {
  datasetName: string;
  actor: string;
  judge?: boolean;
  k?: number;
  model?: string | null;
}

export interface EvalCaseResult {
  id: string;
  question: string;
  tags: string[];
  status: 'ok' | 'refused' | 'error';
  error: string | null;
  queryId: string | null;
  latencyMs: number;
  retrieved: { chunkId: number; source: string; title: string; score: number; via: string[] }[];
  citations: number;
  expectedSources: string[];
  matchedSources: string[];
  hit: boolean | null;
  recall: number | null;
  mrr: number | null;
  citationPrecision: number | null;
  correctness: number | null;
  faithfulness: number | null;
  judgeReasoning: string | null;
  answer: string | null;
}

export interface EvalMetrics {
  cases: number;
  answered: number;
  refused: number;
  errors: number;
  refusalRate: number;
  errorRate: number;
  hitRate: number | null;
  recallAtK: number | null;
  mrr: number | null;
  citationPrecision: number | null;
  citedRate: number;
  correctness: number | null;
  faithfulness: number | null;
  judged: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  graph: { entities: number; relations: number; catchAllRatio: number; catchAllWarning: boolean };
}

function normSource(s: string): string {
  return s.trim().toLowerCase().replace(/^\.\//, '').replace(/\\/g, '/');
}

export function sourceMatches(retrieved: { source: string; title: string }, expected: string): boolean {
  const e = normSource(expected);
  if (!e) return false;
  const src = normSource(retrieved.source);
  const title = normSource(retrieved.title);
  return src === e || title === e || (src.length > 0 && (src.endsWith(e) || e.endsWith(src))) || title.endsWith(e);
}

function mean(values: (number | null)[]): number | null {
  const xs = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (xs.length === 0) return null;
  return Number((xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(4));
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

function judgeMessages(c: EvalCase, answer: KnowledgeAnswer): ChatMessage[] {
  const context = answer.sources.slice(0, 8).map((s, i) => `[${i + 1}] ${s.text.slice(0, 1200)}`).join('\n\n');
  const system = [
    'You are a strict grader for a retrieval-augmented question answering system.',
    'Score two things from 0.0 to 1.0:',
    '- correctness: does the answer correctly and completely answer the question? Use the reference answer when given; otherwise judge from the sources.',
    '- faithfulness: is every claim in the answer supported by the provided sources (no invented facts)?',
    'Return ONLY JSON: {"correctness": 0.0, "faithfulness": 0.0, "reasoning": "one or two sentences"}',
  ].join('\n');
  const user = [
    `Question: ${c.question}`,
    c.expected_answer ? `Reference answer: ${c.expected_answer}` : 'Reference answer: (none provided)',
    `Sources:\n${context || '(none)'}`,
    `Answer to grade:\n${answer.answer}`,
  ].join('\n\n');
  return [{ role: 'system', content: system }, { role: 'user', content: user }];
}

async function judge(c: EvalCase, answer: KnowledgeAnswer, model: string | null | undefined): Promise<{ correctness: number | null; faithfulness: number | null; reasoning: string | null }> {
  try {
    const reply = await knowledgeChat(judgeMessages(c, answer), { temperature: 0, maxTokens: 400, model: model ?? null, waitForResetSeconds: 180 });
    const parsed = extractJsonObject(reply.text) as { correctness?: unknown; faithfulness?: unknown; reasoning?: unknown } | null;
    const clamp = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : null);
    return {
      correctness: clamp(parsed?.correctness),
      faithfulness: clamp(parsed?.faithfulness),
      reasoning: typeof parsed?.reasoning === 'string' ? parsed.reasoning.slice(0, 500) : null,
    };
  } catch (err: any) {
    return { correctness: null, faithfulness: null, reasoning: `judge failed: ${String(err?.message ?? err).slice(0, 160)}` };
  }
}

export async function evaluateCase(base: BaseRow, c: EvalCase, opts: EvalOptions): Promise<EvalCaseResult> {
  const started = Date.now();
  const expected = c.expected_sources ?? [];
  const result: EvalCaseResult = {
    id: c.id, question: c.question, tags: c.tags ?? [], status: 'ok', error: null, queryId: null, latencyMs: 0, retrieved: [], citations: 0,
    expectedSources: expected, matchedSources: [], hit: null, recall: null, mrr: null, citationPrecision: null, correctness: null, faithfulness: null, judgeReasoning: null, answer: null,
  };
  const scoreRetrieval = (sources: { source: string; documentTitle: string }[]) => {
    if (expected.length === 0) return;
    const matched = new Set<string>();
    let firstRank: number | null = null;
    sources.forEach((s, i) => {
      for (const e of expected) {
        if (sourceMatches({ source: s.source, title: s.documentTitle }, e)) {
          matched.add(e);
          if (firstRank == null) firstRank = i + 1;
        }
      }
    });
    result.matchedSources = [...matched];
    result.hit = matched.size > 0;
    result.recall = matched.size / expected.length;
    result.mrr = firstRank == null ? 0 : 1 / firstRank;
  };

  let answer: KnowledgeAnswer;
  try {
    answer = await answerQuestion(base, c.question, { actor: opts.actor, kind: 'eval', maxChunks: opts.k, model: opts.model ?? null });
  } catch (err: any) {
    result.status = 'error';
    result.error = String(err?.message ?? err).slice(0, 300);
    result.latencyMs = Date.now() - started;
    // Generation failed but retrieval ran: score what we have, so retrieval
    // quality is measurable even when no model is available.
    const sources = (err?.knowledgeSources ?? []) as KnowledgeAnswer['sources'];
    result.retrieved = sources.map(s => ({ chunkId: s.chunkId, source: s.source, title: s.documentTitle, score: s.score, via: s.via }));
    scoreRetrieval(sources);
    return result;
  }
  result.queryId = answer.id;
  result.status = answer.status;
  result.answer = answer.answer;
  result.latencyMs = answer.latencyMs;
  result.retrieved = answer.sources.map(s => ({ chunkId: s.chunkId, source: s.source, title: s.documentTitle, score: s.score, via: s.via }));
  result.citations = answer.citations.length;
  scoreRetrieval(answer.sources);
  if (expected.length > 0 && answer.citations.length > 0) {
    const good = answer.citations.filter(cit => expected.some(e => sourceMatches({ source: cit.source, title: cit.title }, e))).length;
    result.citationPrecision = good / answer.citations.length;
  }

  if (opts.judge && answer.status === 'ok') {
    const j = await judge(c, answer, opts.model);
    result.correctness = j.correctness;
    result.faithfulness = j.faithfulness;
    result.judgeReasoning = j.reasoning;
  }
  return result;
}

export function aggregate(results: EvalCaseResult[], base: BaseRow): EvalMetrics {
  const policy = getPolicy();
  const g = graphStats(base.id);
  const latencies = results.map(r => r.latencyMs);
  const answered = results.filter(r => r.status === 'ok').length;
  const refused = results.filter(r => r.status === 'refused').length;
  const errors = results.filter(r => r.status === 'error').length;
  const withExpected = results.filter(r => r.hit != null);
  const judged = results.filter(r => r.correctness != null || r.faithfulness != null).length;
  return {
    cases: results.length,
    answered, refused, errors,
    refusalRate: results.length ? Number((refused / results.length).toFixed(4)) : 0,
    errorRate: results.length ? Number((errors / results.length).toFixed(4)) : 0,
    hitRate: withExpected.length ? Number((withExpected.filter(r => r.hit).length / withExpected.length).toFixed(4)) : null,
    recallAtK: mean(withExpected.map(r => r.recall)),
    mrr: mean(withExpected.map(r => r.mrr)),
    citationPrecision: mean(results.map(r => r.citationPrecision)),
    citedRate: answered ? Number((results.filter(r => r.status === 'ok' && r.citations > 0).length / answered).toFixed(4)) : 0,
    correctness: mean(results.map(r => r.correctness)),
    faithfulness: mean(results.map(r => r.faithfulness)),
    judged,
    latencyP50Ms: percentile(latencies, 0.5),
    latencyP95Ms: percentile(latencies, 0.95),
    graph: { entities: g.entities, relations: g.relations, catchAllRatio: Number(g.catchAllRatio.toFixed(4)), catchAllWarning: g.relations > 0 && g.catchAllRatio > policy.graph.catch_all_warn_ratio },
  };
}

export async function runEval(base: BaseRow, cases: EvalCase[], opts: EvalOptions): Promise<EvalRunRow> {
  const id = crypto.randomUUID();
  const config = { judge: !!opts.judge, k: opts.k ?? getPolicy().retrieval.max_chunks, model: opts.model ?? null, policyVersion: getPolicy().versionTag };
  insertEvalRun({ id, base_id: base.id, dataset: opts.datasetName, cases: cases.length, config_json: JSON.stringify(config) });
  audit({ action: 'eval.start', actor: opts.actor, baseId: base.id, target: `eval:${id}`, details: { dataset: opts.datasetName, cases: cases.length, judge: !!opts.judge } });
  const results: EvalCaseResult[] = [];
  try {
    for (const c of cases) results.push(await evaluateCase(base, c, opts));
    const metrics = aggregate(results, base);
    finishEvalRun(id, 'done', metrics, results);
    audit({ action: 'eval.done', actor: opts.actor, baseId: base.id, target: `eval:${id}`, details: { hitRate: metrics.hitRate, correctness: metrics.correctness, faithfulness: metrics.faithfulness, refusalRate: metrics.refusalRate } });
  } catch (err: any) {
    const message = String(err?.message ?? err).slice(0, 300);
    finishEvalRun(id, 'error', results.length ? aggregate(results, base) : {}, results, message);
    audit({ action: 'eval.error', actor: opts.actor, baseId: base.id, target: `eval:${id}`, details: { error: message } });
  }
  return getEvalRun(id)!;
}

const pct = (v: number | null) => (v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`);

export function evalReportMarkdown(run: EvalRunRow, baseSlug: string): string {
  const m = JSON.parse(run.metrics_json || '{}') as Partial<EvalMetrics>;
  const results = JSON.parse(run.results_json || '[]') as EvalCaseResult[];
  const lines: string[] = [];
  lines.push(`# Knowledge eval — ${run.dataset} on \`${baseSlug}\``);
  lines.push('');
  lines.push(`Run \`${run.id}\` · ${run.status} · ${new Date(run.started_at_ms).toISOString()}${run.error ? ` · error: ${run.error}` : ''}`);
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('| --- | --- |');
  lines.push(`| Cases | ${m.cases ?? results.length} (answered ${m.answered ?? 0}, refused ${m.refused ?? 0}, errors ${m.errors ?? 0}) |`);
  lines.push(`| Retrieval hit rate | ${pct(m.hitRate ?? null)} |`);
  lines.push(`| Recall@k | ${pct(m.recallAtK ?? null)} |`);
  lines.push(`| MRR | ${m.mrr == null ? 'n/a' : m.mrr.toFixed(3)} |`);
  lines.push(`| Citation precision | ${pct(m.citationPrecision ?? null)} |`);
  lines.push(`| Answers with citations | ${pct(m.citedRate ?? null)} |`);
  lines.push(`| Correctness (judge) | ${pct(m.correctness ?? null)}${m.judged ? ` over ${m.judged}` : ''} |`);
  lines.push(`| Faithfulness (judge) | ${pct(m.faithfulness ?? null)} |`);
  lines.push(`| Refusal rate | ${pct(m.refusalRate ?? null)} |`);
  lines.push(`| Latency p50 / p95 | ${m.latencyP50Ms ?? 0} ms / ${m.latencyP95Ms ?? 0} ms |`);
  if (m.graph) lines.push(`| Graph | ${m.graph.entities} entities, ${m.graph.relations} relations, RELATED_TO share ${pct(m.graph.catchAllRatio)}${m.graph.catchAllWarning ? ' ⚠ ontology needs a more specific relation' : ''} |`);
  lines.push('');
  lines.push('## Cases');
  lines.push('');
  lines.push('| Case | Status | Hit | Recall | MRR | Cit. prec. | Correct | Faithful | ms |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const r of results) {
    lines.push(`| ${r.id} | ${r.status} | ${r.hit == null ? '-' : r.hit ? 'yes' : 'no'} | ${pct(r.recall)} | ${r.mrr == null ? '-' : r.mrr.toFixed(2)} | ${pct(r.citationPrecision)} | ${pct(r.correctness)} | ${pct(r.faithfulness)} | ${r.latencyMs} |`);
  }
  const misses = results.filter(r => r.hit === false || r.status === 'error' || (r.correctness != null && r.correctness < 0.5));
  if (misses.length) {
    lines.push('');
    lines.push('## Needs attention');
    lines.push('');
    for (const r of misses) {
      lines.push(`- **${r.id}** (${r.status}): ${r.question}`);
      if (r.error) lines.push(`  - error: ${r.error}`);
      if (r.hit === false) lines.push(`  - expected ${r.expectedSources.join(', ')}; retrieved ${[...new Set(r.retrieved.map(x => x.source || x.title))].slice(0, 5).join(', ') || 'nothing'}`);
      if (r.judgeReasoning) lines.push(`  - judge: ${r.judgeReasoning}`);
    }
  }
  return lines.join('\n');
}
