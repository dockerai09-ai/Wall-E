// Recursive text splitter: paragraphs first, then sentences, then a hard
// character split, with a word-aligned overlap between neighbours so a fact
// that straddles a boundary is still retrievable from at least one chunk.
// Token counts are estimated at ~4 characters per token, which is what the
// rest of the server uses for budgeting (lib/token-estimate does the same).

export interface TextChunk {
  ordinal: number;
  text: string;
  /** Offsets of the chunk's own content in the source (overlap excluded). */
  charStart: number;
  charEnd: number;
  tokenCount: number;
}

export interface ChunkOptions {
  targetTokens?: number;
  overlapTokens?: number;
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

interface Piece { text: string; start: number; end: number }

function splitParagraphs(text: string): Piece[] {
  const out: Piece[] = [];
  const re = /\n[ \t]*\n+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    pushPiece(out, text, last, m.index);
    last = m.index + m[0].length;
  }
  pushPiece(out, text, last, text.length);
  return out;
}

function pushPiece(out: Piece[], text: string, start: number, end: number): void {
  // Trim whitespace but keep offsets pointing at the retained content.
  let s = start;
  let e = end;
  while (s < e && /\s/.test(text[s])) s++;
  while (e > s && /\s/.test(text[e - 1])) e--;
  if (e > s) out.push({ text: text.slice(s, e), start: s, end: e });
}

function splitSentences(piece: Piece): Piece[] {
  const out: Piece[] = [];
  const re = /(?<=[.!?。！？])\s+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(piece.text)) !== null) {
    pushPiece(out, piece.text, last, m.index);
    last = m.index + m[0].length;
  }
  pushPiece(out, piece.text, last, piece.text.length);
  return out.map(p => ({ text: p.text, start: piece.start + p.start, end: piece.start + p.end }));
}

function hardSplit(piece: Piece, maxChars: number): Piece[] {
  const out: Piece[] = [];
  let pos = 0;
  while (pos < piece.text.length) {
    let end = Math.min(piece.text.length, pos + maxChars);
    if (end < piece.text.length) {
      // Prefer breaking at whitespace inside the last 20% of the window.
      const ws = piece.text.lastIndexOf(' ', end);
      if (ws > pos + maxChars * 0.8) end = ws;
    }
    out.push({ text: piece.text.slice(pos, end), start: piece.start + pos, end: piece.start + end });
    pos = end;
    while (pos < piece.text.length && /\s/.test(piece.text[pos])) pos++;
  }
  return out;
}

/** Break the source into pieces none of which exceed maxChars. */
function atomize(text: string, maxChars: number): Piece[] {
  const pieces: Piece[] = [];
  for (const para of splitParagraphs(text)) {
    if (para.text.length <= maxChars) { pieces.push(para); continue; }
    for (const sentence of splitSentences(para)) {
      if (sentence.text.length <= maxChars) pieces.push(sentence);
      else pieces.push(...hardSplit(sentence, maxChars));
    }
  }
  return pieces;
}

function tailWords(text: string, maxChars: number): string {
  if (maxChars <= 0 || text.length <= maxChars) return text.length <= maxChars ? text : '';
  const cut = text.length - maxChars;
  const ws = text.indexOf(' ', cut);
  return ws === -1 ? '' : text.slice(ws + 1);
}

export function chunkText(source: string, opts: ChunkOptions = {}): TextChunk[] {
  const targetTokens = Math.max(50, opts.targetTokens ?? 400);
  const overlapTokens = Math.max(0, Math.min(opts.overlapTokens ?? 60, Math.floor(targetTokens / 2)));
  const text = source.replace(/\r\n?/g, '\n');
  const targetChars = targetTokens * 4;
  const overlapChars = overlapTokens * 4;

  const pieces = atomize(text, targetChars);
  const chunks: TextChunk[] = [];
  let current: Piece[] = [];
  let currentLen = 0;
  let prevText = '';

  const flush = () => {
    if (current.length === 0) return;
    const body = current.map(p => p.text).join('\n\n');
    const overlap = tailWords(prevText, overlapChars);
    const full = overlap ? `${overlap}\n\n${body}` : body;
    chunks.push({
      ordinal: chunks.length,
      text: full,
      charStart: current[0].start,
      charEnd: current[current.length - 1].end,
      tokenCount: estimateTokens(full),
    });
    prevText = body;
    current = [];
    currentLen = 0;
  };

  for (const piece of pieces) {
    const extra = piece.text.length + (current.length ? 2 : 0);
    if (current.length && currentLen + extra > targetChars) flush();
    current.push(piece);
    currentLen += extra;
  }
  flush();
  return chunks;
}
