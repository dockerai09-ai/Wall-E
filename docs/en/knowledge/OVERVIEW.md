**English** · [简体中文](../../zh-cn/knowledge/OVERVIEW.md)

[← Back to README](../../../README.md) · [Documentation index](../README.md)

# Knowledge domain overview

The knowledge module turns Wall-E from a router into a grounded assistant over your own documents. One page covers it end to end:

- **[01 — Knowledge module](01-knowledge-module.md)**: quick start, pipeline, embedders, ontology and graph, Neo4j, governance, provenance, evals, API and CLI reference, limitations.
- **[02 — Clinical speech-to-text (Swedish)](02-clinical-stt-swedish.md)**: Wall-E vs Claude Pro for transcribing patient speech, the benchmark (WER, term recall, PII leaks), the Claude Pro capture protocol, and the loop that optimises the pipeline against it.

Related code: `server/src/services/knowledge/` (pipeline), `server/src/routes/knowledge.ts` and `routes/rag.ts` (HTTP), `server/src/db/migrations/20260909_051719_knowledge_store.ts` (schema), `knowledge/` at the repo root (ontology, governance policy, eval datasets); `server/src/services/stt/` and `routes/stt.ts` for speech-to-text, `knowledge/stt/<profile>/` for glossaries, benchmark cases and loop configuration.
