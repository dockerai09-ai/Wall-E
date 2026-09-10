**English** · [简体中文](../../zh-cn/knowledge/01-knowledge-module.md)

[← Back to README](../../../README.md) · [Documentation index](../README.md) · [Knowledge overview](OVERVIEW.md)

# Knowledge module: RAG, knowledge graph, evals, governance, provenance

Wall-E can answer questions from **your own documents** with citations, backed by a hybrid vector + keyword index, an ontology-constrained knowledge graph (mirrored into Neo4j when you have one), a governance policy that is enforced on every request, a provenance record for every answer, and an evaluation harness so retrieval and answer quality are measured rather than assumed.

Everything runs inside the existing server: same SQLite database, same providers, same unified/profile keys, same dashboard session. Nothing new has to be deployed to get started; Neo4j is optional.

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Embedders](#embedders)
- [Ontology and the knowledge graph](#ontology-and-the-knowledge-graph)
- [Neo4j](#neo4j)
- [Governance](#governance)
- [Provenance](#provenance)
- [Evals](#evals)
- [API reference](#api-reference)
- [CLI](#cli)
- [Limitations](#limitations)

## Quick start

With the server running and at least one chat provider key added on the **Keys** page:

```bash
# 1. Load Wall-E's own documentation into a base called "docs" and index it
npm run kb:ingest -- --base docs --dir docs/en --source-prefix docs/en/ --index

# 2. Ask, with your unified API key
curl -s http://localhost:3001/v1/rag/query \
  -H "Authorization: Bearer $FREELLMAPI_KEY" -H "Content-Type: application/json" \
  -d '{"knowledge_base":"docs","question":"How do I rotate the encryption key?"}' | jq .answer

# 3. Measure it
npm run kb:eval -- --base docs --dataset wall-e-docs --judge
```

The answer carries `citations` (which numbered source backs which claim), the `sources` themselves, the graph entities that were traversed, the model used, and a `provenance` link to the full lineage record.

The same operations are available from the dashboard session on `/api/knowledge/*` (see the [API reference](#api-reference)).

## How it works

```
ingest ──► redact PII ──► hash + dedupe ──► chunk ──► SQLite (document, chunks)
                                                      │
              background indexer ◄────────────────────┘
              ├─ embed chunks (router / OpenAI-compatible / hash)
              └─ extract entities + relations per chunk (LLM, ontology-constrained)
                     └─ SQLite graph tables ──► Neo4j mirror (optional)

query ──► redact ──► vector top-k ─┐
                     BM25 (FTS5) ──┼─► reciprocal-rank fusion ──► graph expansion ──► context
                                   │   (entity linking → k-hop → evidence chunks)
                                   ▼
                     prompt with numbered sources ──► router (fallback chain) ──► answer + [n] citations
                                                                                  └─► provenance record + audit row
```

**Retrieval** is hybrid by default: cosine similarity over chunk embeddings and BM25 over an FTS5 index, fused with reciprocal-rank fusion so a chunk that both methods like ranks first. Scores are normalised to the best hit (1.0) and `retrieval.min_score` in the policy is a fraction of that.

**Graph expansion** links entity names in the question (plus the entities the top hits mention) to graph nodes, walks up to `max_hops` relations, and pulls in the chunks that evidence those relations. Chunks reached only through the graph enter just below the weakest direct hit; direct hits the graph also supports get a bonus. Every hit records how it got there (`via: vector | keyword | graph`).

**Generation** goes through the normal fallback chain (`services/router.ts`) with the same accounting as any proxied request, tagged `knowledge` in analytics. The prompt contains only numbered sources and the question; the model is told to cite with `[n]` and to reply with a fixed refusal sentence when the sources do not contain the answer. With `generation.refuse_without_evidence: true` (the default) an empty retrieval is refused before any model is called.

## Configuration

Environment variables (all documented in `.env.example`; see [the variable reference](../env/01-variables.md#knowledge)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `KB_ENABLED` | `true` | Master switch for `/api/knowledge` and `/v1/rag`. |
| `KB_EMBEDDER` | `auto` | `auto`, `router`, `openai` or `hash`, see [Embedders](#embedders). |
| `KB_EMBEDDING_MODEL` | unset | Model for the `router`/`openai` embedders. |
| `KB_EMBEDDING_BASE_URL`, `KB_EMBEDDING_API_KEY` | unset | Endpoint for the `openai` embedder (Ollama, OpenAI, Jina, TEI…). |
| `KB_ONTOLOGY_PATH` | `knowledge/ontology.yaml` | Ontology file. |
| `KB_GOVERNANCE_PATH` | `knowledge/governance.yaml` | Governance policy file. |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | unset | Neo4j mirror; leave `NEO4J_URI` empty to stay on SQLite. |

Both YAML files are read once at first use and re-read with `POST /api/knowledge/ontology/reload` and `POST /api/knowledge/governance/reload`; each reload is audited with the new version hash.

## Embedders

The embedder is chosen **once per knowledge base** when the base is created, so all of its chunks live in one vector space. It is recorded on the base and in every provenance record.

| `KB_EMBEDDER` | What it uses | When to pick it |
| --- | --- | --- |
| `router` | Wall-E's own `/v1/embeddings` engine: any provider enabled on the **Embeddings** page, with the engine's within-family failover. | You already have an embedding provider key (Gemini, NVIDIA, Cloudflare, Cohere, custom…). |
| `openai` | Any OpenAI-compatible `/embeddings` endpoint. Point it at Ollama (`http://127.0.0.1:11434/v1`, model `nomic-embed-text`) for fully local embeddings. | Local-only setups, or a dedicated embedding service. |
| `hash` | Deterministic feature hashing (word uni/bigrams + character trigrams, 256 dims). No network, no keys, no model download. | Tests, air-gapped smoke runs. Retrieval quality is well below a learned model; the FTS5 keyword leg does most of the work. |
| `auto` (default) | Probes `router` once; falls back to `hash` with a logged warning if no embedding provider is usable. | The default: works out of the box, upgrades itself when you add a provider. |

`POST /api/knowledge/bases/:ref/reindex` clears embeddings and graph state so a base can be re-embedded after changing the embedder.

## Ontology and the knowledge graph

`knowledge/ontology.yaml` declares the **entity classes** (`Person`, `Organization`, `Product`, `Technology`, `Concept`, `Event`, `Location` by default) and the **relation types** with their permitted endpoints (`WORKS_FOR: Person → Organization`, `USES: Product|Organization|Person|Technology → Technology|Product`, …, and a `RELATED_TO: any → any` catch-all).

During indexing every chunk is sent to the fallback chain (or the model pinned by `graph.extraction_model`) with the ontology in the prompt. The reply is validated: unknown classes, relation types, or endpoint classes are dropped and counted, confidence below `graph.min_confidence` is dropped, names are canonicalised (lower-cased, whitespace-collapsed) so `Neo4j` and `neo4j` merge into one node. SQLite stores entities, chunk→entity mentions, and one relation row **per evidencing chunk**, so every edge can be traced back to text.

Eval reports show the share of `RELATED_TO` edges; when it exceeds `graph.catch_all_warn_ratio` the ontology is missing a relation your corpus needs. Add it and reindex.

Graph extraction costs one model call per chunk. Turn it off with `graph.extract: false` in the policy for a pure vector + keyword base.

## Neo4j

Set `NEO4J_URI` (plus user/password) and Wall-E mirrors the graph into Neo4j: `(:KnowledgeBase)-[:HAS_DOCUMENT]->(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)`, entities labelled with their ontology class (`:Entity:Technology`), relations typed from the ontology and carrying `chunkKey` and `confidence`, plus the ontology itself as `(:Ontology)-[:HAS_CLASS]->(:OntologyClass)` / `-[:HAS_RELATION]->(:OntologyRelation)` with uniqueness constraints. Keys are derived from the SQLite ids (`kb<base>:e<entity>`), so the two stores reconcile trivially.

When Neo4j is reachable, retrieval traverses it (`graph.engine: "neo4j"` in responses); when it is down, traversal falls back to SQLite recursive CTEs and ingest keeps working. Connection failures are retried every 60 s and never break a request.

With the bundled compose file:

```bash
podman compose --profile graph up -d      # or: docker compose --profile graph up -d
# then in .env:  NEO4J_URI=bolt://neo4j:7687  NEO4J_PASSWORD=<what you set>
```

Neo4j Browser is on http://localhost:7474. `POST /api/knowledge/graph/cypher` runs read-only Cypher from the dashboard session (write statements are rejected by the driver's READ access mode).

## Governance

`knowledge/governance.yaml` is enforced by the server, not documented advice:

| Section | Enforced where | Highlights |
| --- | --- | --- |
| `ingest` | before chunking | PII redaction (email, phone, credit card with Luhn, IBAN, SSN, API keys, IPs, plus your own named regexes), size cap, allowed content types, content-hash dedupe, chunk sizing. Redacted text is what gets stored, embedded and sent to models; the original never lands in the database. |
| `retrieval` | every search/query | hybrid on/off, max chunks, score floor, graph expansion hops/limits. |
| `generation` | every answer | refuse without evidence, require citations (flagged as `citationsMissing` in the answer's `governance` block), model allow/deny lists (`groq/`, `openai/gpt-4o`, `model-id`), context budget, temperature, `wait_for_reset_seconds` (wait out a rate-limit window instead of failing; evals always wait). |
| `graph` | indexing | extraction on/off, pinned extraction model, confidence floor, `extractions_per_minute` (default 20, so a single free-tier key is not benched by a burst of extraction calls). |
| `retention` | hourly sweep | age limits for provenance and audit rows, cap on stored eval runs. |
| `access` | `/v1/rag` | bases created by a client-profile key are visible only to that profile unless `shared`; deletes can be disabled. |

Every enforcement (rejection, redaction counts, refusal, deletion, policy or ontology reload, eval run) writes a row to `knowledge_audit` with the actor (`user:<id>`, `unified`, `profile:<id>`, `cli:…`) and the policy version hash. `GET /api/knowledge/audit` reads it.

**Right to be forgotten.** `DELETE /api/knowledge/documents/:id` removes the chunks, embeddings, mentions and evidenced relations (and the Neo4j mirror), prunes entities nothing supports any more, and leaves a tombstone so provenance records that cited the document say `status: "deleted"` instead of dangling.

## Provenance

Every question, answered or refused, gets a `knowledge_queries` row keyed by a UUID (the `id` in the answer):

- the question and its hash, the actor, the kind (`query`, `eval`);
- the retrieval set with per-chunk scores, ranks and `via`;
- the graph context (engine, seed entities, traversed nodes and edges);
- the exact prompt hash, the platform/model that answered, token usage, latency;
- the answer, its hash, the parsed citations;
- the governance block: policy version, ontology hash, redaction count, refused/citations-missing flags, embedder.

`GET /api/knowledge/provenance/:id` returns the record joined with the **current** state of the documents it used (title, source, content hash, `deleted`). `GET /api/knowledge/provenance/:id/prov` renders it as [W3C PROV-JSON](https://www.w3.org/Submission/prov-json/): the answer `wasGeneratedBy` a generation activity `wasAssociatedWith` the model agent, `wasDerivedFrom` the cited chunks, which `wasDerivedFrom` their documents, with the policy as a `used` entity.

## Evals

Datasets are JSONL files in `knowledge/evals/` (or inline cases in the API call):

```json
{"id":"install-port","question":"Which port does the dashboard listen on?","expected_answer":"Port 3001.","expected_sources":["docs/en/install/01-install.md"],"tags":["install"]}
```

Each case runs through the full pipeline (retrieval, generation, provenance) and is scored:

| Metric | Needs | Meaning |
| --- | --- | --- |
| hit rate, recall@k, MRR | `expected_sources` | Did retrieval surface the right documents, and how high. Sources match by path suffix or title. |
| citation precision | `expected_sources` | Share of the answer's citations that point at expected sources. |
| cited rate, refusal rate, error rate | – | Behavioural counters. |
| correctness, faithfulness | `--judge` | An LLM judge (through the router) scores 0–1 against the reference answer and the sources. Optional, so retrieval-only runs cost no model calls. |
| latency p50/p95 | – | End-to-end per case. |
| graph catch-all share | – | `RELATED_TO` share of edges; a warning means the ontology needs a more specific relation. |

Runs are stored (`GET /api/knowledge/evals`) so quality can be compared across ontology, policy, embedder or model changes; `GET /api/knowledge/evals/:id/report` renders a Markdown report with a "needs attention" list. `knowledge/evals/wall-e-docs.jsonl` ships as a self-referential smoke test over Wall-E's own docs.

## API reference

Dashboard session (`Authorization: Bearer <session>`), all under `/api/knowledge`:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/status` | Config, ontology/policy versions, Neo4j and indexer state, FTS availability. |
| GET/POST | `/bases` | List / create (`{name, slug?, description?, shared?}`; the embedder is chosen here). |
| GET/DELETE | `/bases/:ref` | Base with stats / delete (cascades). `:ref` is an id or slug. |
| POST | `/bases/:ref/documents` | Ingest `{title?, text}` or `{url}` (http(s) only; HTML is converted). Returns 201, or 200 with `deduplicated: true`. |
| GET | `/bases/:ref/documents` | List (`?includeDeleted=1` for tombstones). |
| GET/DELETE | `/documents/:id` | Document (`?chunks=1` includes chunks) / forget it. |
| POST | `/bases/:ref/reindex` | Clear embeddings + graph state and re-run. |
| POST | `/bases/:ref/index-now` | One synchronous indexer pass (returns counts). |
| POST | `/bases/:ref/search` | Retrieval only: `{question, maxChunks?, graph?, hybrid?}`. |
| POST | `/bases/:ref/query` | Full answer: `{question, maxChunks?, model?, graph?, temperature?}`. |
| GET | `/bases/:ref/queries` | Recent provenance summaries. |
| GET | `/provenance/:id`, `/provenance/:id/prov` | Provenance record / PROV-JSON. |
| GET | `/bases/:ref/graph/stats`, `/graph/entities?q=&class=`, `/graph/entities/:id` | Graph inspection. |
| POST | `/graph/cypher` | Read-only Cypher against Neo4j (`503` when not connected). |
| GET/POST | `/ontology`, `/ontology/reload`, `/ontology/sync` | Ontology; reload also syncs Neo4j. |
| GET/POST | `/governance`, `/governance/reload` | Policy. |
| GET | `/audit?baseId=&action=&limit=` | Audit log. |
| GET/POST | `/evals`, `/evals/run`, `/evals/datasets`, `/evals/:id`, `/evals/:id/report` | Eval runs (`{base, dataset | cases, judge?, k?, model?}`). |

API keys (unified or client-profile key, like every other `/v1` route):

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/rag/bases` | Bases visible to this key. |
| POST | `/v1/rag/query` | `{knowledge_base, question, max_chunks?, model?, graph?, temperature?}` → `knowledge.answer` with citations, sources, graph, model, usage, provenance link. A profile's enforced system prompt is prepended. |
| POST | `/v1/rag/search` | Retrieval only. |
| POST | `/v1/rag/documents` | Add text to a base from an agent or app. |

Errors use the usual `{ "error": { "message", "type" } }` shape: `400` validation, `404` unknown base/document, `409` duplicate slug/content, `413` over the size cap, `415` disallowed content type, `503` no model could answer or Neo4j unavailable.

## CLI

```bash
npm run kb:ingest -- --base <slug> --dir <path> [--name <name>] [--ext md,txt,html] [--source-prefix <p>] [--index]
npm run kb:eval   -- --base <slug> --dataset <name|file.jsonl> [--judge] [--k 12] [--model platform/model] [--out report.md] [--json]
```

Both read `.env` / `FREEAPI_DB_PATH` like the server and are safe to run alongside it (WAL mode).

## Limitations

- The vector search is brute force over one base's embeddings (cached, decoded once). It is fast to roughly 100k chunks; beyond that, swap `services/knowledge/vector.ts` for `sqlite-vec` or an external store.
- The `hash` embedder is a fallback, not a retrieval model. Use `router` or `openai` for real corpora.
- Graph extraction quality is the extraction model's quality. Free-tier models sometimes return prose instead of JSON; such chunks are marked `graph_status: error` and the document still becomes ready.
- PDF, DOCX and images are not parsed; convert them to text/Markdown first (or add a converter in `services/knowledge/ingest.ts`).
- Ingest-by-URL is available only to the dashboard session, never to API keys.
- Node's built-in SQLite (the Android install path) may lack FTS5; retrieval then degrades to vector + term counting and `GET /api/knowledge/status` reports `fts: false`.
