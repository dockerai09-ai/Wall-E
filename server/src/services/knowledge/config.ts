import path from 'path';
import { fileURLToPath } from 'url';

// Knowledge module configuration, read from the environment on every call so
// tests can flip variables between cases. Every variable is documented in
// .env.example under "Knowledge"; lib/env-drift.ts treats those as known.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// server/src/services/knowledge -> repo root (same depth from server/dist).
export const REPO_ROOT = path.resolve(__dirname, '../../../..');

export type EmbedderKind = 'auto' | 'router' | 'openai' | 'hash';

export interface Neo4jConfig {
  uri: string;
  user: string;
  password: string;
  database: string;
}

export interface KnowledgeConfig {
  enabled: boolean;
  embedder: EmbedderKind;
  embeddingModel: string | null;
  embeddingBaseUrl: string | null;
  embeddingApiKey: string | null;
  ontologyPath: string;
  governancePath: string;
  neo4j: Neo4jConfig | null;
}

function str(name: string): string | null {
  const raw = process.env[name];
  if (raw === undefined) return null;
  const trimmed = raw.trim();
  return trimmed === '' ? null : trimmed;
}

function bool(name: string, fallback: boolean): boolean {
  const raw = str(name)?.toLowerCase();
  if (raw == null) return fallback;
  if (raw === 'true' || raw === '1' || raw === 'yes' || raw === 'on') return true;
  if (raw === 'false' || raw === '0' || raw === 'no' || raw === 'off') return false;
  return fallback;
}

export function getKnowledgeConfig(): KnowledgeConfig {
  const embedderRaw = (str('KB_EMBEDDER') ?? 'auto').toLowerCase();
  const embedder: EmbedderKind = embedderRaw === 'router' || embedderRaw === 'openai' || embedderRaw === 'hash'
    ? embedderRaw
    : 'auto';
  const neo4jUri = str('NEO4J_URI');
  return {
    enabled: bool('KB_ENABLED', true),
    embedder,
    embeddingModel: str('KB_EMBEDDING_MODEL'),
    embeddingBaseUrl: str('KB_EMBEDDING_BASE_URL'),
    embeddingApiKey: str('KB_EMBEDDING_API_KEY'),
    ontologyPath: str('KB_ONTOLOGY_PATH') ?? path.join(REPO_ROOT, 'knowledge', 'ontology.yaml'),
    governancePath: str('KB_GOVERNANCE_PATH') ?? path.join(REPO_ROOT, 'knowledge', 'governance.yaml'),
    neo4j: neo4jUri
      ? {
          uri: neo4jUri,
          user: str('NEO4J_USER') ?? 'neo4j',
          password: str('NEO4J_PASSWORD') ?? '',
          database: str('NEO4J_DATABASE') ?? 'neo4j',
        }
      : null,
  };
}

/** Errors raised by the knowledge module carry an HTTP status so routes can
 *  map them 1:1 without a translation table. */
export class KnowledgeError extends Error {
  status: number;
  type: string;
  constructor(message: string, status = 400, type = 'invalid_request_error') {
    super(message);
    this.name = 'KnowledgeError';
    this.status = status;
    this.type = type;
  }
}
