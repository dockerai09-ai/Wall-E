#!/usr/bin/env node
/**
 * kb-eval — run an evaluation dataset against a knowledge base and print the
 * markdown report. Retrieval metrics need no model; --judge grades answers
 * with an LLM through the router.
 *
 * Usage:
 *   tsx src/scripts/kb-eval.ts --base <slug> --dataset <name|path.jsonl> [--judge] [--k 12]
 *                              [--model platform/model_id] [--out report.md] [--json]
 *
 * Exit status: 0 when the run completed, 1 on a run-level error.
 */
import '../env.js';
import fs from 'fs';
import path from 'path';
import { initDb } from '../db/index.js';
import { REPO_ROOT } from '../services/knowledge/config.js';
import { evalReportMarkdown, parseJsonl, runEval } from '../services/knowledge/evals.js';
import { resolveBase } from '../services/knowledge/store.js';

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const v = process.argv[i + 1];
  return v && !v.startsWith('--') ? v : fallback;
}
const flag = (name: string) => process.argv.includes(`--${name}`);
// `npm run ... -w server` runs in server/; resolve user paths against the directory npm was invoked from.
const userCwd = process.env.INIT_CWD || process.cwd();

async function main(): Promise<void> {
  const slug = arg('base');
  const dataset = arg('dataset');
  if (!slug || !dataset) {
    console.error('usage: kb-eval --base <slug> --dataset <name|path.jsonl> [--judge] [--k 12] [--model p/m] [--out report.md] [--json]');
    process.exit(1);
  }
  const userFile = path.resolve(userCwd, dataset);
  const file = dataset.endsWith('.jsonl') && fs.existsSync(userFile)
    ? userFile
    : path.join(REPO_ROOT, 'knowledge', 'evals', `${dataset.replace(/\.jsonl$/, '')}.jsonl`);
  if (!fs.existsSync(file)) {
    console.error(`dataset not found: ${file}`);
    process.exit(1);
  }

  initDb(process.env.FREEAPI_DB_PATH?.trim() || undefined);
  const base = resolveBase(slug);
  if (!base) {
    console.error(`knowledge base '${slug}' not found`);
    process.exit(1);
  }
  const cases = parseJsonl(fs.readFileSync(file, 'utf8'));
  console.error(`running ${cases.length} cases from ${path.basename(file)} against '${base.slug}'${flag('judge') ? ' with LLM judge' : ''}...`);
  const run = await runEval(base, cases, {
    datasetName: path.basename(file, '.jsonl'), actor: 'cli:kb-eval', judge: flag('judge'),
    k: arg('k') ? Number(arg('k')) : undefined, model: arg('model') ?? null,
  });
  const report = evalReportMarkdown(run, base.slug);
  if (flag('json')) console.log(JSON.stringify({ id: run.id, status: run.status, metrics: JSON.parse(run.metrics_json), results: JSON.parse(run.results_json) }, null, 2));
  else console.log(report);
  const out = arg('out');
  if (out) {
    const outPath = path.resolve(userCwd, out);
    fs.writeFileSync(outPath, report);
    console.error(`report written to ${outPath}`);
  }
  process.exit(run.status === 'done' ? 0 : 1);
}

main().catch(err => {
  console.error(err?.stack ?? err);
  process.exit(1);
});
