#!/usr/bin/env node
/**
 * kb-ingest — bulk-load a directory of text/markdown/html files into a
 * knowledge base, optionally running the indexer to completion.
 *
 * Usage:
 *   tsx src/scripts/kb-ingest.ts --base <slug> --dir <path> [--name <display name>]
 *                                [--ext md,mdx,txt,html,json,csv] [--source-prefix <prefix>]
 *                                [--index] [--max-ticks 500]
 *
 *   --base    slug of the knowledge base; created (with the configured embedder) when missing
 *   --dir     directory to walk recursively
 *   --ext     comma-separated extensions to include (default: md,mdx,txt,html,htm,json,csv)
 *   --source-prefix  prepended to each file's relative path in `source` (default: none)
 *   --index   run the indexer until nothing is pending (embeddings + graph extraction)
 *
 * Reads the same .env / FREEAPI_DB_PATH as the server; run with the server
 * stopped or accept that both write to the same SQLite file (WAL makes that safe).
 */
import '../env.js';
import fs from 'fs';
import path from 'path';
import { initDb } from '../db/index.js';
import { chooseEmbedder } from '../services/knowledge/embedder.js';
import { ingestDocument } from '../services/knowledge/ingest.js';
import { runIndexerOnce } from '../services/knowledge/indexer.js';
import { basesWithPendingWork, createBase, resolveBase } from '../services/knowledge/store.js';

const CONTENT_TYPES: Record<string, string> = {
  md: 'text/markdown', mdx: 'text/markdown', txt: 'text/plain', html: 'text/html', htm: 'text/html', json: 'application/json', csv: 'text/csv',
};

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const v = process.argv[i + 1];
  return v && !v.startsWith('--') ? v : fallback;
}
const flag = (name: string) => process.argv.includes(`--${name}`);
// `npm run ... -w server` runs in server/; resolve user paths against the directory npm was invoked from.
const userCwd = process.env.INIT_CWD || process.cwd();

function walk(dir: string, exts: Set<string>, out: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, exts, out);
    else if (exts.has(path.extname(entry.name).slice(1).toLowerCase())) out.push(full);
  }
  return out;
}

function firstHeading(text: string): string | null {
  const m = /^#\s+(.+)$/m.exec(text);
  return m ? m[1].trim() : null;
}

async function main(): Promise<void> {
  const slug = arg('base');
  const dir = arg('dir');
  if (!slug || !dir) {
    console.error('usage: kb-ingest --base <slug> --dir <path> [--name <name>] [--ext md,txt] [--source-prefix <p>] [--index]');
    process.exit(1);
  }
  const root = path.resolve(userCwd, dir);
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    console.error(`not a directory: ${root}`);
    process.exit(1);
  }
  const exts = new Set((arg('ext', 'md,mdx,txt,html,htm,json,csv') as string).split(',').map(s => s.trim().toLowerCase()).filter(Boolean));
  const prefix = arg('source-prefix', '') as string;

  initDb(process.env.FREEAPI_DB_PATH?.trim() || undefined);

  let base = resolveBase(slug);
  if (!base) {
    const choice = await chooseEmbedder();
    base = createBase({ slug, name: arg('name', slug) as string, embedder: choice.embedder.kind, embeddingModel: choice.embedder.model });
    console.log(`created base '${base.slug}' (embedder ${choice.embedder.model})${choice.warning ? `\n  ! ${choice.warning}` : ''}`);
  } else {
    console.log(`using base '${base.slug}' (embedder ${base.embedding_model || base.embedder})`);
  }

  const files = walk(root, exts);
  let added = 0;
  let deduped = 0;
  let failed = 0;
  let chunks = 0;
  for (const file of files) {
    const rel = path.relative(root, file).split(path.sep).join('/');
    const text = fs.readFileSync(file, 'utf8');
    const ext = path.extname(file).slice(1).toLowerCase();
    try {
      const r = await ingestDocument(base, {
        title: firstHeading(text) ?? rel, text, source: prefix + rel, contentType: CONTENT_TYPES[ext] ?? 'text/plain', actor: 'cli:kb-ingest',
        metadata: { file: rel, bytes: Buffer.byteLength(text) },
      });
      if (r.deduplicated) deduped++; else { added++; chunks += r.chunks; }
      process.stdout.write(`${r.deduplicated ? '=' : '+'} ${rel} (${r.chunks} chunks${r.redactions ? `, ${r.redactions} redactions` : ''})\n`);
    } catch (err: any) {
      failed++;
      process.stdout.write(`! ${rel}: ${err?.message ?? err}\n`);
    }
  }
  console.log(`\n${added} added (${chunks} chunks), ${deduped} unchanged, ${failed} failed, from ${files.length} files`);

  if (flag('index')) {
    const maxTicks = Number(arg('max-ticks', '500'));
    let ticks = 0;
    while (basesWithPendingWork().includes(base.id) && ticks < maxTicks) {
      const r = await runIndexerOnce({ embedLimit: 64, graphLimit: 8 });
      ticks++;
      if (r.embedded || r.extracted || r.documentsReady) console.log(`tick ${ticks}: embedded ${r.embedded}, extracted ${r.extracted}, ready ${r.documentsReady}`);
      if (r.errors.length) console.warn(r.errors.slice(0, 3).map(e => `  ! ${e}`).join('\n'));
      if (!r.embedded && !r.extracted && !r.documentsReady && r.errors.length) break;
    }
    console.log(basesWithPendingWork().includes(base.id) ? 'indexing stopped with work pending (see errors above)' : 'indexing complete');
  } else {
    console.log('run again with --index, or let the running server\'s indexer pick the documents up');
  }
}

main().catch(err => {
  console.error(err?.stack ?? err);
  process.exit(1);
});
