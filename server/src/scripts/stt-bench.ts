#!/usr/bin/env node
/**
 * stt-bench — score speech-to-text arms on a profile's benchmark.
 *
 *   tsx src/scripts/stt-bench.ts [--profile sv-medical] [--arm wall-e]
 *       [--model auto|whisper-large-v3] [--prompt glossary|none] [--correction llm|off]
 *       [--correction-model p/m] [--no-redact] [--limit N] [--tags pii,läkemedel]
 *       [--rpm 18] [--no-cache] [--out report.md] [--json]
 *   tsx src/scripts/stt-bench.ts --import arms/claude-pro.jsonl --arm claude-pro
 *   tsx src/scripts/stt-bench.ts --compare          # table of the latest run per arm
 *
 * Runs are stored in stt_runs (metrics and per-case scores; never transcripts).
 * Exit status 0 when every case was scored, 1 otherwise.
 */
import '../env.js';
import fs from 'fs';
import path from 'path';
import { initDb } from '../db/index.js';
import { loadManifest, filterCases, runArm, scoreManualArm, parseArmFile, persistRun, benchReportMarkdown, FileCache, metricsTable, runFromRow } from '../services/stt/bench.js';
import { parseVariant, variantKey } from '../services/stt/pipeline.js';
import { loadGlossary, profileDir } from '../services/stt/glossary.js';
import { listRuns } from '../services/stt/store.js';

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const v = process.argv[i + 1];
  return v && !v.startsWith('--') ? v : fallback;
}
const flag = (name: string) => process.argv.includes(`--${name}`);
const userCwd = process.env.INIT_CWD || process.cwd();

async function main(): Promise<void> {
  initDb(process.env.FREEAPI_DB_PATH?.trim() || undefined);
  const profile = arg('profile', 'sv-medical')!;
  const dir = profileDir(profile);

  if (flag('compare')) {
    const byArm = new Map<string, ReturnType<typeof runFromRow>>();
    for (const row of listRuns({ profile, limit: 500 })) {
      if (row.status !== 'done') continue;
      const arm = row.kind === 'loop' ? `${row.arm} (loop ${row.loop_id})` : row.arm;
      if (!byArm.has(arm)) byArm.set(arm, runFromRow(row));
    }
    if (byArm.size === 0) { console.log('no runs yet'); return; }
    console.log(metricsTable([...byArm.entries()].map(([label, r]) => ({ label: `${label} #${r.id}${r.variant ? ' `' + variantKey(r.variant) + '`' : ''}`, metrics: r.metrics }))));
    return;
  }

  const manifestFile = path.join(dir, 'audio', 'manifest.jsonl');
  const cases = fs.existsSync(manifestFile)
    ? loadManifest(manifestFile)
    : loadManifest(path.join(dir, 'cases.jsonl'));
  const selected = filterCases(cases, {
    limit: arg('limit') ? Number(arg('limit')) : undefined,
    tags: arg('tags')?.split(',').map(s => s.trim()).filter(Boolean),
    ids: arg('ids')?.split(',').map(s => s.trim()).filter(Boolean),
  });

  const importFile = arg('import');
  let run;
  if (importFile) {
    const arm = arg('arm', 'claude-pro')!;
    const file = path.resolve(userCwd, importFile);
    run = scoreManualArm(selected, parseArmFile(fs.readFileSync(file, 'utf8')), profile, arm);
    console.error(`scored ${run.metrics.scored}/${run.metrics.cases} captured transcripts for arm '${arm}'`);
  } else {
    if (!fs.existsSync(manifestFile)) {
      console.error(`no synthesized audio: run ${path.join(dir, 'synth.sh')} first`);
      process.exit(1);
    }
    const variant = parseVariant({
      sttModel: arg('model', 'auto'),
      prompt: arg('prompt', 'glossary'),
      correction: arg('correction', 'llm'),
      correctionModel: arg('correction-model') ?? null,
      temperature: Number(arg('temperature', '0')),
      redact: !flag('no-redact'),
      structure: false,
    });
    const arm = arg('arm', 'wall-e')!;
    const glossary = loadGlossary(profile);
    const cache = flag('no-cache') ? undefined : new FileCache(path.join(dir, '.cache'));
    console.error(`running ${selected.length} cases as '${arm}' with ${variantKey(variant)} (glossary ${glossary.hash})...`);
    run = await runArm(selected, variant, {
      profile, arm, glossary, cache, rpm: Number(arg('rpm', '18')),
      onProgress: (d, t, r) => console.error(`  ${d}/${t} ${r.id} ${r.error ? 'ERROR ' + r.error : `wer=${r.score?.wer.toFixed(3)}${r.cached ? ' (cached)' : ''}`}`),
    });
    if (cache) console.error(`cache: ${cache.hits} hits, ${cache.misses} misses`);
  }
  const row = persistRun(run, { kind: 'bench' });
  const report = benchReportMarkdown([run], { title: `STT benchmark — ${profile} — ${run.arm} (run #${row.id})` });
  if (flag('json')) console.log(JSON.stringify({ id: row.id, arm: run.arm, variant: run.variant, metrics: run.metrics, cases: run.cases }, null, 2));
  else console.log(report);
  const out = arg('out');
  if (out) {
    const outPath = path.resolve(userCwd, out);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, report);
    console.error(`report written to ${outPath}`);
  }
  process.exit(run.metrics.errors === 0 ? 0 : 1);
}

main().catch(err => {
  console.error(err?.stack ?? err);
  process.exit(1);
});
