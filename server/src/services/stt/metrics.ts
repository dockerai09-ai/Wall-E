// Scoring for Swedish clinical transcripts. Everything here is deterministic
// and provider-free so a benchmark run and a unit test score identically.
//
// Normalisation is applied to reference and hypothesis alike before any
// comparison, so a recogniser that writes "50 mg" for a patient who said
// "femtio milligram" is not penalised: both sides become "50 mg". The
// normaliser handles Swedish number words (compounds included), decimal
// "komma", dose units, a few spelling variants Whisper alternates between
// (igår/i går), and collapses spoken digit groups so a personnummer read as
// "85 04 12 12 34" compares equal to "850412-1234".

export interface AlignOp {
  op: 'ok' | 'sub' | 'del' | 'ins';
  ref?: string;
  hyp?: string;
}

export interface CaseScore {
  id: string;
  refTokens: number;
  wer: number;
  cer: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  termsTotal: number;
  termsFound: number;
  termRecall: number;
  piiTotal: number;
  piiLeaked: number;
  /** Reference words the hypothesis got wrong (substituted or dropped). */
  missed: string[];
}

const UNITS: Record<string, number> = {
  noll: 0, en: 1, ett: 1, två: 2, tva: 2, tre: 3, fyra: 4, fem: 5, sex: 6, sju: 7, åtta: 8, atta: 8, nio: 9,
  tio: 10, elva: 11, tolv: 12, tretton: 13, fjorton: 14, femton: 15, sexton: 16, sjutton: 17, arton: 18, nitton: 19,
};
const TENS: Record<string, number> = {
  tjugo: 20, trettio: 30, fyrtio: 40, femtio: 50, sextio: 60, sjuttio: 70, åttio: 80, attio: 80, nittio: 90,
};

/** Parse one Swedish number word (possibly a compound such as "tjugofem",
 *  "etthundrafemtio", "tvåtusen"). Returns null when the token is not a number. */
export function parseSwedishNumber(token: string): number | null {
  const t = token.toLowerCase();
  if (/^\d+([.,]\d+)?$/.test(t)) return Number(t.replace(',', '.'));
  if (t in UNITS) return UNITS[t];
  if (t in TENS) return TENS[t];
  let rest = t;
  let total = 0;
  let matched = false;
  // thousands
  const thousand = /^(.*?)tusen(.*)$/.exec(rest);
  if (thousand) {
    const head = thousand[1] === '' || thousand[1] === 'ett' || thousand[1] === 'en' ? 1 : parseSwedishNumber(thousand[1]);
    if (head == null) return null;
    total += head * 1000;
    rest = thousand[2];
    matched = true;
  }
  const hundred = /^(.*?)hundra(.*)$/.exec(rest);
  if (hundred) {
    const head = hundred[1] === '' || hundred[1] === 'ett' || hundred[1] === 'en' ? 1 : parseSwedishNumber(hundred[1]);
    if (head == null || head >= 10) return null;
    total += head * 100;
    rest = hundred[2];
    matched = true;
  }
  if (rest === '') return matched ? total : null;
  if (rest in UNITS) return total + UNITS[rest];
  if (rest in TENS) return total + TENS[rest];
  for (const [word, value] of Object.entries(TENS)) {
    if (rest.startsWith(word)) {
      const unit = rest.slice(word.length);
      if (unit in UNITS && UNITS[unit] < 10) return total + value + UNITS[unit];
    }
  }
  return null;
}

const UNIT_WORDS: Record<string, string> = {
  milligram: 'mg', milligrams: 'mg', mikrogram: 'mcg', mikrogrammet: 'mcg', µg: 'mcg', ug: 'mcg',
  gram: 'g', kilo: 'kg', kilogram: 'kg', milliliter: 'ml', millilitrar: 'ml', liter: 'l', procent: '%',
  centimeter: 'cm', millimeter: 'mm', kilometer: 'km',
};

const SPELLING: Record<string, string> = {
  'i går': 'igår', 'i dag': 'idag', 'i morgon': 'imorgon', 'i kväll': 'ikväll', 'i natt': 'inatt', 'i morse': 'imorse',
  'i stället': 'istället', 'skall': 'ska', 'sen': 'sedan', 'nån': 'någon', 'nåt': 'något', 'dom': 'de',
  'i förrgår': 'iförrgår', 'sån': 'sådan', 'mej': 'mig', 'dej': 'dig', 'e-post': 'epost', 'okej': 'ok',
};

/** Normalise Swedish text for comparison. Returns tokens. */
export function normalizeSwedish(text: string): string[] {
  let t = text.toLowerCase();
  // Join number-word phrases first: "tolv komma fem" -> "12.5"
  t = t.replace(/[–—]/g, '-');
  for (const [from, to] of Object.entries(SPELLING)) t = t.replace(new RegExp(`\\b${from}\\b`, 'g'), to);
  // redaction markers are not words; the PII they replaced is masked out of
  // both sides in scoreCase, so they must vanish here rather than count
  t = t.replace(/\[redacted:[a-z_]+\]/g, ' ');
  // strip punctuation except a decimal separator between digits
  t = t.replace(/(\d)[.,](\d)|[^\p{L}\p{N}\s%-]/gu, (m, a: string | undefined, b: string | undefined) => (a !== undefined ? `${a}.${b}` : ' '));
  t = t.replace(/(\d)-(\d)/g, '$1 $2');
  t = t.replace(/-/g, ' ');
  let tokens = t.split(/\s+/).filter(Boolean);
  // number words -> digits, decimal "komma", units
  const out: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    // digits stay verbatim (leading zeros matter in identifiers); only number
    // WORDS are converted
    if (/^\d+(\.\d+)?$/.test(tok)) {
      if (tokens[i + 1] === 'komma' && tokens[i + 2] != null && /^\d+$/.test(tokens[i + 2])) { out.push(`${tok}.${tokens[i + 2]}`); i += 2; continue; }
      out.push(tok);
      continue;
    }
    const n = parseSwedishNumber(tok);
    if (n != null) {
      // "tolv komma fem" / "12 komma 5"
      if (tokens[i + 1] === 'komma') {
        const frac = tokens[i + 2] != null ? parseSwedishNumber(tokens[i + 2]) : null;
        if (frac != null) { out.push(`${n}.${frac}`); i += 2; continue; }
      }
      out.push(String(n));
      continue;
    }
    if (tok in UNIT_WORDS) { out.push(UNIT_WORDS[tok]); continue; }
    // "50mg" -> "50 mg"
    const m = /^(\d+(?:\.\d+)?)(mg|mcg|g|kg|ml|l|%)$/.exec(tok);
    if (m) { out.push(m[1], m[2]); continue; }
    out.push(tok);
  }
  tokens = out;
  // Collapse runs of small digit groups (spoken digits) into one number so a
  // personnummer or phone number read digit-by-digit compares equal to the
  // compact form: "85 04 12 12 34" -> "8504121234".
  const collapsed: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    if (/^\d{1,6}$/.test(tokens[i])) {
      let j = i;
      let run = '';
      while (j < tokens.length && /^\d{1,6}$/.test(tokens[j])) { run += tokens[j]; j++; }
      // an identifier: several groups whose digits add up to a personnummer
      // (10/12) or a Swedish phone number (8-10 digits starting with 0)
      const identifier = j - i >= 2 && (run.length === 10 || run.length === 12 || (/^0/.test(run) && run.length >= 8 && run.length <= 10));
      if (identifier) { collapsed.push(run); i = j - 1; continue; }
    }
    collapsed.push(tokens[i]);
  }
  return collapsed;
}

export function normalizeSwedishText(text: string): string {
  return normalizeSwedish(text).join(' ');
}

/** Levenshtein alignment with backtrace, on token arrays (or characters). */
export function align<T>(ref: T[], hyp: T[], eq: (a: T, b: T) => boolean = (a, b) => a === b): AlignOp[] {
  const n = ref.length;
  const m = hyp.length;
  const d: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = 0; i <= n; i++) d[i][0] = i;
  for (let j = 0; j <= m; j++) d[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = eq(ref[i - 1], hyp[j - 1]) ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
    }
  }
  const ops: AlignOp[] = [];
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (eq(ref[i - 1], hyp[j - 1]) ? 0 : 1)) {
      ops.push({ op: eq(ref[i - 1], hyp[j - 1]) ? 'ok' : 'sub', ref: String(ref[i - 1]), hyp: String(hyp[j - 1]) });
      i--; j--;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      ops.push({ op: 'del', ref: String(ref[i - 1]) });
      i--;
    } else {
      ops.push({ op: 'ins', hyp: String(hyp[j - 1]) });
      j--;
    }
  }
  return ops.reverse();
}

export interface EditCounts { substitutions: number; deletions: number; insertions: number; refLength: number }

export function countEdits(ops: AlignOp[], refLength: number): EditCounts {
  let substitutions = 0;
  let deletions = 0;
  let insertions = 0;
  for (const o of ops) {
    if (o.op === 'sub') substitutions++;
    else if (o.op === 'del') deletions++;
    else if (o.op === 'ins') insertions++;
  }
  return { substitutions, deletions, insertions, refLength };
}

export function errorRate(c: EditCounts): number {
  if (c.refLength === 0) return c.insertions > 0 ? 1 : 0;
  return (c.substitutions + c.deletions + c.insertions) / c.refLength;
}

/** Word error rate between two raw texts (normalised first). */
export function wer(reference: string, hypothesis: string): number {
  const r = normalizeSwedish(reference);
  const h = normalizeSwedish(hypothesis);
  return errorRate(countEdits(align(r, h), r.length));
}

/** Character error rate on the normalised text without spaces. */
export function cer(reference: string, hypothesis: string): number {
  const r = [...normalizeSwedish(reference).join('')];
  const h = [...normalizeSwedish(hypothesis).join('')];
  return errorRate(countEdits(align(r, h), r.length));
}

/** A term counts as found when its normalised tokens occur contiguously. */
export function containsTerm(hypothesisTokens: string[], term: string): boolean {
  const t = normalizeSwedish(term);
  if (t.length === 0) return true;
  outer: for (let i = 0; i + t.length <= hypothesisTokens.length; i++) {
    for (let k = 0; k < t.length; k++) if (hypothesisTokens[i + k] !== t[k]) continue outer;
    return true;
  }
  return false;
}

/** PII leaks when the digit string of a protected value appears in the
 *  digit stream of the output, whatever the spacing. */
export function leaksPii(text: string, value: string): boolean {
  const needle = value.replace(/\D/g, '');
  if (needle.length < 6) return normalizeSwedishText(text).includes(normalizeSwedishText(value));
  const hay = normalizeSwedish(text).join(' ').replace(/\D/g, '');
  return hay.includes(needle);
}

export interface ScoreInput {
  id: string;
  reference: string;
  hypothesis: string;
  terms?: string[];
  pii?: string[];
}

/** Drop the tokens that carry a protected value so redaction (a governance
 *  choice) neither helps nor hurts the accuracy metrics; leaks are counted
 *  separately by piiLeakRate. */
export function maskPii(tokens: string[], pii: string[]): string[] {
  if (pii.length === 0) return tokens;
  const digits = pii.map(v => v.replace(/\D/g, '')).filter(d => d.length >= 6);
  const words = pii.filter(v => v.replace(/\D/g, '').length < 6).map(v => normalizeSwedish(v).join(' '));
  let out = tokens.filter(t => !digits.some(d => t.replace(/\D/g, '') === d));
  for (const w of words) {
    if (!w) continue;
    const joined = out.join(' ').replace(new RegExp(`(^| )${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?= |$)`, 'g'), '$1');
    out = joined.split(' ').filter(Boolean);
  }
  return out;
}

export function scoreCase(input: ScoreInput): CaseScore {
  const piiValues = input.pii ?? [];
  const r = maskPii(normalizeSwedish(input.reference), piiValues);
  const h = maskPii(normalizeSwedish(input.hypothesis), piiValues);
  const ops = align(r, h);
  const edits = countEdits(ops, r.length);
  const terms = input.terms ?? [];
  const found = terms.filter(t => containsTerm(h, t)).length;
  const pii = input.pii ?? [];
  const leaked = pii.filter(v => leaksPii(input.hypothesis, v)).length;
  const missed = ops.filter(o => (o.op === 'sub' || o.op === 'del') && o.ref).map(o => o.ref as string);
  return {
    id: input.id,
    refTokens: r.length,
    wer: errorRate(edits),
    cer: errorRate(countEdits(align([...r.join('')], [...h.join('')]), r.join('').length)),
    substitutions: edits.substitutions,
    deletions: edits.deletions,
    insertions: edits.insertions,
    termsTotal: terms.length,
    termsFound: found,
    termRecall: terms.length === 0 ? 1 : found / terms.length,
    piiTotal: pii.length,
    piiLeaked: leaked,
    missed,
  };
}

export interface AggregateMetrics {
  cases: number;
  scored: number;
  errors: number;
  /** Corpus-level WER: total edits / total reference words. */
  wer: number;
  cer: number;
  /** Mean of per-case WER (each case weighs the same). */
  werMean: number;
  termRecall: number;
  piiLeakRate: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  /** Cases whose LLM correction was rejected by the hallucination guard. */
  correctionRejections: number;
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

export function aggregate(
  scores: CaseScore[],
  extra: { errors?: number; latenciesMs?: number[]; correctionRejections?: number; totalCases?: number } = {},
): AggregateMetrics {
  const refWords = scores.reduce((n, s) => n + s.refTokens, 0);
  const edits = scores.reduce((n, s) => n + s.substitutions + s.deletions + s.insertions, 0);
  const termsTotal = scores.reduce((n, s) => n + s.termsTotal, 0);
  const termsFound = scores.reduce((n, s) => n + s.termsFound, 0);
  const piiTotal = scores.reduce((n, s) => n + s.piiTotal, 0);
  const piiLeaked = scores.reduce((n, s) => n + s.piiLeaked, 0);
  const cerWeighted = scores.reduce((n, s) => n + s.cer * s.refTokens, 0);
  const lat = extra.latenciesMs ?? [];
  return {
    cases: extra.totalCases ?? scores.length + (extra.errors ?? 0),
    scored: scores.length,
    errors: extra.errors ?? 0,
    wer: refWords === 0 ? 0 : round(edits / refWords),
    cer: refWords === 0 ? 0 : round(cerWeighted / refWords),
    werMean: scores.length === 0 ? 0 : round(scores.reduce((n, s) => n + s.wer, 0) / scores.length),
    termRecall: termsTotal === 0 ? 1 : round(termsFound / termsTotal),
    piiLeakRate: piiTotal === 0 ? 0 : round(piiLeaked / piiTotal),
    latencyP50Ms: Math.round(percentile(lat, 50)),
    latencyP95Ms: Math.round(percentile(lat, 95)),
    correctionRejections: extra.correctionRejections ?? 0,
  };
}

function round(n: number): number {
  return Math.round(n * 10000) / 10000;
}

const STOPWORDS = new Set([
  'och', 'att', 'det', 'jag', 'har', 'är', 'en', 'ett', 'i', 'på', 'som', 'med', 'för', 'av', 'till', 'den', 'inte',
  'om', 'så', 'men', 'de', 'mig', 'min', 'mitt', 'mina', 'han', 'hon', 'sig', 'kan', 'när', 'vid', 'ut', 'upp', 'ner',
  'var', 'blir', 'sedan', 'igår', 'idag', 'efter', 'innan', 'från', 'än', 'hela', 'också', 'eller', 'utan', 'över',
]);

/** Reference words that were substituted or dropped in at least
 *  `minOccurrences` cases, ordered by frequency. Numbers, stopwords and short
 *  tokens are skipped: they are not vocabulary the recogniser can be primed with. */
export function frequentMisses(scores: CaseScore[], minOccurrences = 2, max = 40): { term: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const s of scores) {
    for (const w of new Set(s.missed)) {
      if (w.length < 4 || STOPWORDS.has(w) || /^\d/.test(w)) continue;
      counts.set(w, (counts.get(w) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .filter(([, c]) => c >= minOccurrences)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, max)
    .map(([term, count]) => ({ term, count }));
}
