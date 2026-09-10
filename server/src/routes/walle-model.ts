// /v1/chat/completions handler for the virtual `wall-e/<profile>` models
// (see services/stt/assistant.ts) and their /v1/models discovery entries.
// The proxy route hands over right after schema validation, so the whole
// OpenAI wire shape (messages with input_audio parts, stream, temperature,
// max_tokens) is exactly what any client already sends.

import crypto from 'crypto';
import type { Request, Response } from 'express';
import type { ChatMessage } from '@freellmapi/shared/types.js';
import type { ResolvedAuth } from '../lib/system-prompt.js';
import { KnowledgeError } from '../services/knowledge/config.js';
import { MediaError } from '../services/media.js';
import { listWallEModels, runWallEAssistant, type WallEAssistant } from '../services/stt/assistant.js';

interface WireBody {
  model?: string;
  messages: { role: string; content?: unknown; name?: string }[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
}

export function wallEModelEntries(autoContextWindow: number | null, available: boolean) {
  return listWallEModels().map(a => ({
    id: a.id,
    object: 'model' as const,
    created: 0,
    // same owner tag and reason vocabulary as every other router entry, so
    // clients that filter on them keep working
    owned_by: 'freellmapi',
    name: a.name,
    description: a.description,
    context_window: autoContextWindow,
    context_length: autoContextWindow,
    available,
    unavailable_reason: available ? null : 'no_key',
  }));
}

export async function handleWallEChatCompletion(req: Request, res: Response, auth: ResolvedAuth, body: WireBody, assistant: WallEAssistant): Promise<void> {
  const messages: ChatMessage[] = body.messages
    .filter(m => m.role === 'system' || m.role === 'developer' || m.role === 'user' || m.role === 'assistant')
    .map(m => ({ role: m.role === 'developer' ? 'system' : (m.role as ChatMessage['role']), content: (m.content ?? '') as ChatMessage['content'] }));
  // a client's own system message is advice to the persona, not a replacement
  const clientSystem = messages.filter(m => m.role === 'system').map(m => (typeof m.content === 'string' ? m.content : '')).filter(Boolean).join('\n');
  const enforced = [auth.kind === 'profile' ? auth.systemPrompt : null, clientSystem || null].filter(Boolean).join('\n\n') || null;
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;

  let result;
  try {
    result = await runWallEAssistant(assistant, {
      messages: messages.filter(m => m.role !== 'system'),
      actor: auth.kind === 'unified' ? 'unified' : `profile:${auth.profileId}`,
      enforcedSystemPrompt: enforced,
      temperature: body.temperature,
      maxTokens: maxTokens && maxTokens > 0 ? maxTokens : undefined,
    });
  } catch (err: any) {
    const status = err instanceof KnowledgeError ? err.status : err instanceof MediaError ? err.status : 502;
    const httpStatus = status >= 400 && status < 600 ? status : 502;
    res.status(httpStatus).json({ error: { message: `${assistant.id}: ${err?.message ?? 'unknown error'}`, type: httpStatus >= 500 ? 'api_error' : 'invalid_request_error' } });
    return;
  }

  const id = `chatcmpl-${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);
  const extension = {
    mode: result.mode,
    profile: assistant.profile,
    assistant_hash: assistant.hash,
    transcripts: result.transcripts,
    answer: result.answer,
    redactions: result.redactions,
    generation_model: result.model,
    latency_ms: result.latencyMs,
  };
  res.setHeader('X-Provider', 'wall-e');
  res.setHeader('X-Model', assistant.id);

  if (!body.stream) {
    res.json({
      id, object: 'chat.completion', created, model: assistant.id,
      choices: [{ index: 0, message: { role: 'assistant', content: result.content }, finish_reason: 'stop' }],
      usage: result.usage,
      wall_e: extension,
    });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();
  const frame = (delta: Record<string, unknown>, finish: string | null = null, extra: Record<string, unknown> = {}) => {
    try {
      res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created, model: assistant.id, choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`);
    } catch { /* socket gone */ }
  };
  frame({ role: 'assistant', content: '' });
  // paragraph-sized chunks: the text is already complete, this only keeps
  // streaming clients rendering progressively
  for (const piece of result.content.split(/(?<=\n\n)/)) if (piece) frame({ content: piece });
  frame({}, 'stop', { usage: result.usage, wall_e: extension });
  try { res.write('data: [DONE]\n\n'); res.end(); } catch { /* socket gone */ }
}
