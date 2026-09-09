// One chat completion through the router with the same accounting the proxy
// and fusion paths do (request counts, token usage, cooldowns, request log),
// tagged 'knowledge' in requested_model so analytics can attribute it. Shape
// mirrors services/fusion.ts runModelCall on purpose: failover across keys and
// models, non-retryable errors stop, governance may veto a routed model.
//
// Tests inject a fake with setKnowledgeChatForTests so nothing here needs a
// provider or a key.

import type { ChatMessage, TokenUsage } from '@freellmapi/shared/types.js';
import { getDb } from '../../db/index.js';
import { routeRequest, routePinnedModel, recordRateLimitHit, recordSuccess, type RouteResult } from '../router.js';
import {
  recordRequest, recordTokens, setCooldown, getCooldownDurationForLimit, getCooldownDecisionForLimit,
  getPaymentRequiredCooldownMs, getModelForbiddenCooldownMs,
} from '../ratelimit.js';
import { logRequest } from '../../lib/request-log.js';
import {
  isRetryableError, isRateLimitSignal, isPaymentRequiredError, isModelNotFoundError, isModelAccessForbiddenError,
} from '../../lib/error-classify.js';
import { contentToString } from '../../lib/content.js';
import { sanitizeProviderErrorMessage } from '../../lib/error-redaction.js';
import { KnowledgeError } from './config.js';
import { getPolicy, isModelAllowed } from './governance.js';
import { estimateTokens } from './chunker.js';

export const KNOWLEDGE_TAG = 'knowledge';

export interface ChatOptions {
  temperature?: number;
  maxTokens?: number;
  /** Pin a catalog model: "platform/model_id" or "model_id". null = fallback chain. */
  model?: string | null;
  timeoutMs?: number;
  maxAttempts?: number;
  /** When every route is rate-limited and the router names a reset time, sleep
   *  until then (capped at this many seconds) and try again, at most twice. */
  waitForResetSeconds?: number;
}

const RESET_RE = /reset\s*~?\s*(\d+)\s*s\b/i;

// When the router reports rate limits but names no reset time ("3 rate-limited
// or on cooldown ... failed earlier this request"), wait this long before retrying.
const FALLBACK_WAIT_MS = 15_000;

/** "Soonest reset ~18s" -> 18000; a rate-limit message without a time ->
 *  FALLBACK_WAIT_MS; null when nothing suggests waiting would help. */
export function parseResetMs(message: string | undefined): number | null {
  if (!message) return null;
  const m = RESET_RE.exec(message);
  if (m) return Number(m[1]) * 1000;
  return /rate-limited|on cooldown|rate limit/i.test(message) ? FALLBACK_WAIT_MS : null;
}

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

export interface ChatResult {
  text: string;
  platform: string;
  modelId: string;
  usage: TokenUsage;
}

export type ChatFn = (messages: ChatMessage[], opts: ChatOptions) => Promise<ChatResult>;

let override: ChatFn | null = null;

export function setKnowledgeChatForTests(fn: ChatFn | null): void {
  override = fn;
}

const ZERO_USAGE: TokenUsage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

/** Resolve "platform/model_id" or "model_id" to models.id, enabled rows only. */
export function findModelDbId(ref: string): number | null {
  const trimmed = ref.trim();
  if (!trimmed) return null;
  const slash = trimmed.indexOf('/');
  const db = getDb();
  if (slash > 0) {
    const platform = trimmed.slice(0, slash);
    const modelId = trimmed.slice(slash + 1);
    const row = db.prepare('SELECT id FROM models WHERE platform = ? AND model_id = ? AND enabled = 1').get(platform, modelId) as { id: number } | undefined;
    if (row) return row.id;
  }
  const row = db.prepare('SELECT id FROM models WHERE model_id = ? AND enabled = 1 ORDER BY id LIMIT 1').get(trimmed) as { id: number } | undefined;
  return row?.id ?? null;
}

export async function knowledgeChat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<ChatResult> {
  if (override) return override(messages, opts);
  const budgetMs = Math.max(0, opts.waitForResetSeconds ?? 0) * 1000;
  let waited = 0;
  for (let round = 0; ; round++) {
    try {
      return await knowledgeChatOnce(messages, opts);
    } catch (err: any) {
      const resetMs = err instanceof KnowledgeError && err.status === 503 ? parseResetMs(err.message) : null;
      if (resetMs == null || round >= 3 || waited + resetMs > budgetMs) throw err;
      await sleep(resetMs + 500);
      waited += resetMs + 500;
    }
  }
}

async function knowledgeChatOnce(messages: ChatMessage[], opts: ChatOptions): Promise<ChatResult> {

  const policy = getPolicy();
  const maxTokens = opts.maxTokens ?? 1024;
  const estimated = messages.reduce((n, m) => n + estimateTokens(contentToString(m.content ?? '')), 0) + maxTokens;
  const pinnedId = opts.model ? findModelDbId(opts.model) : null;
  if (opts.model && pinnedId == null) {
    throw new KnowledgeError(`model '${opts.model}' is not an enabled catalog model`, 400);
  }

  const skipKeys = new Set<string>();
  const skipModels = new Set<number>();
  const maxAttempts = opts.maxAttempts ?? 4;
  let lastError: string | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let route: RouteResult | null;
    try {
      route = pinnedId != null
        ? routePinnedModel(pinnedId, estimated, skipKeys)
        : routeRequest(estimated, skipKeys, undefined, false, false, skipModels);
    } catch (err: any) {
      lastError = sanitizeProviderErrorMessage(err?.message);
      break;
    }
    if (!route) break;

    if (!isModelAllowed(route.platform, route.modelId, policy)) {
      route.release?.();
      skipModels.add(route.modelDbId);
      lastError = `model ${route.platform}/${route.modelId} is denied by the governance policy`;
      if (pinnedId != null) break;
      continue;
    }

    const startedAt = Date.now();
    try {
      const result = await route.provider.chatCompletion(route.apiKey, messages, route.modelId, {
        temperature: opts.temperature ?? policy.generation.temperature,
        max_tokens: maxTokens,
        timeoutMs: opts.timeoutMs ?? 90_000,
      });
      const text = contentToString(result.choices?.[0]?.message?.content ?? '');
      if (!text.trim()) {
        logRequest(route.platform, route.modelId, route.keyId, 'error', 0, 0, Date.now() - startedAt, 'empty completion (knowledge)', null, KNOWLEDGE_TAG);
        skipKeys.add(`${route.platform}:${route.modelId}:${route.keyId}`);
        setCooldown(route.platform, route.modelId, route.keyId, getCooldownDurationForLimit(route.platform, route.modelId, route.keyId, { rpd: route.rpdLimit, tpd: route.tpdLimit }));
        recordRateLimitHit(route.modelDbId);
        lastError = `empty completion from ${route.displayName}`;
        continue;
      }
      const usage = result.usage ?? ZERO_USAGE;
      recordRequest(route.platform, route.modelId, route.keyId);
      recordTokens(route.platform, route.modelId, route.keyId, usage.total_tokens ?? 0);
      recordSuccess(route.modelDbId);
      logRequest(route.platform, route.modelId, route.keyId, 'success', usage.prompt_tokens ?? 0, usage.completion_tokens ?? 0, Date.now() - startedAt, null, null, KNOWLEDGE_TAG);
      return { text, platform: route.platform, modelId: route.modelId, usage };
    } catch (err: any) {
      const safe = sanitizeProviderErrorMessage(err?.message);
      logRequest(route.platform, route.modelId, route.keyId, 'error', 0, 0, Date.now() - startedAt, safe, null, KNOWLEDGE_TAG);
      lastError = safe;
      if (isRetryableError(err)) {
        if (isModelNotFoundError(err) || isModelAccessForbiddenError(err)) skipModels.add(route.modelDbId);
        skipKeys.add(`${route.platform}:${route.modelId}:${route.keyId}`);
        const decision = isPaymentRequiredError(err)
          ? { durationMs: getPaymentRequiredCooldownMs(), source: 'credit' as const }
          : isModelAccessForbiddenError(err)
            ? { durationMs: getModelForbiddenCooldownMs(), source: 'tier' as const }
            : getCooldownDecisionForLimit(route.platform, route.modelId, route.keyId, { rpd: route.rpdLimit, tpd: route.tpdLimit }, err.retryAfterMs, { quotaSignal: isRateLimitSignal(err) });
        setCooldown(route.platform, route.modelId, route.keyId, decision.durationMs, decision.source);
        recordRateLimitHit(route.modelDbId);
        continue;
      }
      break;
    } finally {
      route.release?.();
    }
  }

  throw new KnowledgeError(`no model could complete the request: ${lastError ?? 'no available key for model'}`, 503, 'upstream_error');
}

/** Pull the first JSON object out of a model reply (code fences and prose tolerated). */
export function extractJsonObject(text: string): unknown | null {
  const cleaned = text.replace(/```(?:json)?/gi, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end <= start) return null;
  const candidate = cleaned.slice(start, end + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    // Trailing commas are the most common model slip.
    try {
      return JSON.parse(candidate.replace(/,\s*([}\]])/g, '$1'));
    } catch {
      return null;
    }
  }
}
