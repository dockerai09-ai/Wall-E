#!/usr/bin/env node
/**
 * stt-loop — run the loop-engineering optimisation over a profile's pipeline.
 *
 *   tsx src/scripts/stt-loop.ts [--profile sv-medical] [--config loop.yaml]
 *       [--iterations N] [--limit N] [--tags a,b] [--no-cache]
 *
 * Reads knowledge/stt/<profile>/loop.yaml, evaluates candidates on the
 * synthesized benchmark, learns missed vocabulary into glossary.learned.yaml,
 * and writes state.json + report.md under knowledge/stt/<profile>/runs/.
 * Exit status: 0 when the target was met, 2 when the loop converged or ran
 * out of budget without meeting it, 1 on error.
 */
import '../env.js';
import fs from 'fs';
import path from 'path';
import { initDb } from '../db/index.js';
import { loadManifest, filterCases, FileCache } from '../services/stt/bench.js';
import { loadLoopConfig, runLoop } from '../services/stt/loop.js';
import { profileDir } from '../services/stt/glossary.js';

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
  const manifestFile = path.join(dir, 'audio', 'manifest.jsonl');
  if (!fs.existsSync(manifestFile)) {
    console.error(`no synthesized audio: run ${path.join(dir, 'synth.sh')} first`);
    process.exit(1);
  }
  const configFile = arg('config') ? path.resolve(userCwd, arg('config')!) : undefined;
  const config = loadLoopConfig(profile, configFile);
  const cases = filterCases(loadManifest(manifestFile), {
    limit: arg('limit') ? Number(arg('limit')) : undefined,
    tags: arg('tags')?.split(',').map(s => s.trim()).filter(Boolean),
  });
  const state = await runLoop(config, {
    cases,
    cache: flag('no-cache') ? undefined : new FileCache(path.join(dir, '.cache')),
    maxIterations: arg('iterations') ? Number(arg('iterations')) : undefined,
    log: line => console.error(line),
  });
  console.log(fs.readFileSync(path.join(dir, 'runs', `loop-${state.loopId}`, 'report.md'), 'utf8'));
  process.exit(state.status === 'target_met' ? 0 : state.status === 'error' ? 1 : 2);
}

main().catch(err => {
  console.error(err?.stack ?? err);
  process.exit(1);
});
