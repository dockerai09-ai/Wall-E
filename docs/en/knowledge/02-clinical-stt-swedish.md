**English** · [简体中文](../../zh-cn/knowledge/02-clinical-stt-swedish.md)

[← Back to README](../../../README.md) · [Documentation index](../README.md) · [Knowledge module](01-knowledge-module.md)

# Clinical speech-to-text (Swedish): Wall-E vs Claude Pro, the benchmark, and the loop

This page covers one use case end to end: transcribing what a patient says, in Swedish, into text a clinician can use. It compares doing that with Wall-E against doing it with a Claude Pro subscription, describes the benchmark that scores both, and explains the *loop engineering* that keeps improving Wall-E's pipeline against that benchmark.

- [What the two products are](#what-the-two-products-are)
- [Capability comparison](#capability-comparison)
- [Wall-E's clinical STT pipeline](#wall-es-clinical-stt-pipeline)
- [The Wall-E clinical LLM (`wall-e/sv-medical`)](#the-wall-e-clinical-llm-wall-esv-medical)
- [Quick start](#quick-start)
- [The benchmark](#the-benchmark)
- [Capturing the Claude Pro arm](#capturing-the-claude-pro-arm)
- [Loop engineering](#loop-engineering)
- [A local Swedish model (KB-Whisper) through Podman](#a-local-swedish-model-kb-whisper-through-podman)
- [Governance for patient speech](#governance-for-patient-speech)
- [API and CLI reference](#api-and-cli-reference)
- [Limitations and honest caveats](#limitations-and-honest-caveats)

## What the two products are

**Claude Pro** is a chat subscription to claude.ai. Its voice features are consumer dictation and conversation: you speak to the app, the app turns speech into text with a general-purpose recogniser, and Claude replies. There is no audio input in the Claude API, no way to upload a recording, no batch mode, no language- or vocabulary-specific tuning you control, and no deployment option other than Anthropic's cloud. Claude the *language model* is excellent at cleaning up and structuring text, which is the part of the job it can do.

**Wall-E** is a self-hosted router over free-tier model providers and your own endpoints. Its `/v1/audio/transcriptions` endpoint already fails over across Whisper deployments (Groq, Cloudflare Workers AI) and any OpenAI-compatible STT server you register, including a local one. This page adds a *clinical* pipeline on top: glossary-primed recognition, a governed LLM correction pass, redaction of Swedish personal identifiers, optional structuring into journal fields, and hash-only provenance. It runs on your machine or a Podman host, and its accuracy is measured, not assumed.

"Better than Claude Pro" is therefore not one number. It decomposes into questions the benchmark and the comparison table can actually answer: is the transcript more accurate on Swedish clinical speech, does the medical vocabulary survive, does patient-identifying data leak, can it run where patient data must stay, and can you prove what produced a given transcript.

## Capability comparison

| Dimension | Claude Pro (claude.ai app) | Wall-E clinical STT |
|---|---|---|
| Audio input | Live microphone in the app only | Any file (wav, mp3, m4a, flac…) up to 25 MB per request, over HTTP |
| API / batch | None for audio | `POST /v1/stt/transcriptions`, OpenAI-style multipart; CLI batch over a manifest |
| Swedish support | Whatever the app's dictation offers; no control | `language=sv` forced; Swedish-specific normalisation; Swedish glossary; local Swedish models (KB-Whisper) pluggable |
| Medical vocabulary | None | Profile glossary primes the recogniser and the correction pass; the loop learns missed terms |
| Correction of recognition errors | Ask Claude in chat, by hand | Automatic LLM pass, temperature 0, with a hallucination guard that discards rewrites and truncations |
| Personal identifiers (personnummer, phone) | Left in the text | Redacted before the text leaves the pipeline; leak rate is a benchmark metric |
| Where audio and text go | Anthropic's cloud | Your choice: Groq/Cloudflare free tiers, or a local model on the same host (Podman) so nothing leaves the machine |
| What is stored | Conversation history in the app | Hashes, model identities, prompt/glossary/policy versions, redaction counts. No audio, no transcript text |
| Structured output | Ask in chat | Optional journal JSON (kontaktorsak, anamnes, läkemedel, allergier, symtom, oklarheter), schema-validated |
| Measured accuracy | Not measurable programmatically | WER, CER, term recall, PII leak rate, latency on a versioned benchmark, stored per run |
| Improvement over time | Whatever Anthropic ships | Eval-driven loop: coordinate descent over the pipeline knobs plus glossary learning, stopping on a target |
| Cost | Subscription | Free-tier keys, or your own hardware |

The one thing Claude Pro does better is the language model itself: the correction and structuring passes in Wall-E use whatever free models your keys reach, which are weaker than Claude at Swedish nuance. The guard exists precisely because those models sometimes ramble instead of correcting. If you hold an Anthropic API key, Wall-E can route the correction pass to Claude through a custom endpoint and keep everything else.

## Wall-E's clinical STT pipeline

```
audio ─► STT (Wall-E chain: Groq / Cloudflare Whisper, or a registered local endpoint)
        │   primed with the profile glossary as the Whisper prompt (spelling bias)
        ▼
   spoken-number normalisation   "noll sju noll…" → 0701234567, "85 04 12 12 30" → 850412-1230
        ▼
   LLM correction pass (optional) governed model, temperature 0, glossary in the prompt;
        │                          rejected when it changes > 25% of the words or the length drifts
        ▼
   PII redaction                  knowledge/governance.yaml detectors incl. `personnummer` (Luhn-checked)
        ▼
   structuring (optional)         journal JSON, schema-validated, "oklarheter" for what is unclear
        ▼
   provenance row                 sha256 of audio / raw / final, models, prompt hash, glossary hash, policy version
```

Everything that can vary is a field of a **variant** (`sttModel`, `language`, `prompt`, `temperature`, `correction`, `correctionModel`, `redact`, `structure`), so the API, the benchmark and the loop share one code path: `server/src/services/stt/pipeline.ts`.

A **profile** is a directory under `knowledge/stt/<name>/`. The shipped one is `sv-medical`:

| File | Purpose |
|---|---|
| `glossary.yaml` | Curated Swedish clinical vocabulary by category, with common misrecognitions as aliases |
| `glossary.learned.yaml` | Terms the loop found the recogniser missing (generated; safe to edit) |
| `cases.jsonl` | 45 benchmark utterances: reference text, terms that must survive, protected values, tags |
| `synth.sh` | Synthesises the audio with macOS `say` (Swedish voice) or any TTS you point it at |
| `loop.yaml` | Search space, guardrails, target and budget for the loop |
| `arms/` | Hand-captured transcripts for arms that cannot be scripted (Claude Pro) |
| `PROTOCOL.md` | How to capture the Claude Pro arm |
| `assistant.yaml` | What `model: wall-e/sv-medical` does: persona, STT variant, knowledge base, generation settings |
| `audio/`, `.cache/`, `runs/` | Generated: clips, provider-output cache, reports (git-ignored) |

## The Wall-E clinical LLM (`wall-e/sv-medical`)

The pipeline above is also exposed as a **model**: send `model: "wall-e/sv-medical"` to `/v1/chat/completions` and any OpenAI-compatible client (a chat UI, an editor plugin, `curl`, the Wall-E playground) talks to the clinical assistant as if it were one LLM. It is listed in `GET /v1/models` with `owned_by: "wall-e"`. There is one such model per profile directory; its behaviour is declared in `knowledge/stt/<profile>/assistant.yaml` (persona, STT variant, optional knowledge base, generation settings) and every knob has a default, so the file is optional.

It is not a fine-tuned network. It is a composition the router already knows how to run, presented as one id:

| Last user turn contains | Mode | What happens |
|---|---|---|
| audio (`input_audio` part, or a `data:` URL `audio_url`) | document | each clip runs the clinical STT pipeline (glossary-primed recognition, governed correction, redaction, journal-field structuring); the reply is the redacted transcript plus a journal draft. With a text instruction alongside the audio, the persona works on that draft (summarise, translate, list follow-ups…) |
| text only, and `knowledge_base` is set | question | the knowledge module answers with citations from that base, or refuses when the sources do not cover it |
| text only | chat | the persona replies through the router's fallback chain, subject to the governance model policy |

Whatever the mode, the reply is redacted again before it leaves, so a model cannot reintroduce an identifier the transcript step masked; every audio part leaves a hash-only provenance row. The response carries a `wall_e` extension next to the standard `choices`/`usage`: mode, per-part transcripts with structured fields and provenance ids, citations, redaction counts, the generation model. Streaming (`stream: true`) emits standard chunks and ends with the same extension.

```bash
# audio in, journal draft out (OpenAI input_audio shape)
curl -s http://localhost:3001/v1/chat/completions \
  -H "Authorization: Bearer $WALLE_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"wall-e/sv-medical\",\"messages\":[{\"role\":\"user\",\"content\":[
        {\"type\":\"text\",\"text\":\"Lista uppföljningspunkter.\"},
        {\"type\":\"input_audio\",\"input_audio\":{\"data\":\"$(base64 -i besok.wav)\",\"format\":\"wav\"}}]}]}"

# a question, grounded when assistant.yaml names a knowledge base
curl -s http://localhost:3001/v1/chat/completions -H "Authorization: Bearer $WALLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"wall-e/sv-medical","messages":[{"role":"user","content":"Vad gäller vid bröstsmärta?"}]}'
```

Remote audio URLs are refused on purpose: the server never fetches audio from the network on a client's behalf. A client profile key's enforced system prompt and any `system` message the client sends are prepended to the persona, not substituted for it.

## Quick start

```bash
# 1. audio for the benchmark (macOS; see synth.sh for Linux TTS)
npm run stt:synth

# 2. Wall-E arm with the default pipeline
npm run stt:bench -- --arm wall-e

# 3. a generic baseline: plain Whisper, no glossary, no correction, no redaction
npm run stt:bench -- --arm generic-whisper --model whisper-large-v3 --prompt none --correction off --no-redact

# 4. (once) capture and import the Claude Pro arm — see PROTOCOL.md
npm run stt:bench -- --import knowledge/stt/sv-medical/arms/claude-pro.jsonl --arm claude-pro

# 5. compare the latest run per arm
npm run stt:bench -- --compare

# 6. let the loop improve the pipeline against the target
npm run stt:loop
```

Every run lands in `stt_runs` (metrics and per-case scores; never transcripts) and prints a markdown report; `--out` writes it to a file. Use `--limit N`, `--tags pii,läkemedel` or `--ids c01,c02` to run a subset. Provider outputs are cached under `.cache/` keyed by audio hash, model and prompt, so re-scoring after a metric change costs nothing and the loop only pays for knobs it actually changes.

The clinical endpoint itself:

```bash
curl -s http://localhost:3001/v1/stt/transcriptions \
  -H "Authorization: Bearer $WALLE_API_KEY" \
  -F file=@besok.wav -F profile=sv-medical -F structure=true
```

## The benchmark

**Cases.** 45 first-person patient utterances covering cardiology, respiratory, endocrine (diabetes, thyroid), neurology, psychiatry, obstetrics, orthopaedics, dermatology, paediatrics, medication lists with doses, allergies, family and social history, and five cases carrying a personnummer or a phone number. Each case has `reference` (what was said), `terms` (vocabulary that must be recovered verbatim), optional `pii` (values that must not survive), and `tags`.

**Audio.** Synthesised with the macOS Swedish voice so the benchmark is reproducible on any Mac without recording patients. Synthetic speech is cleaner than a consultation room; treat absolute numbers as an upper bound and the *differences between arms* as the signal. To score real recordings, write your own `manifest.jsonl` rows with `audio` paths (consent and the governance policy apply).

**Metrics** (`server/src/services/stt/metrics.ts`), all computed after Swedish normalisation of both sides so that "femtio milligram" and "50 mg", "tolv komma fem" and "12,5", "i går" and "igår" compare equal:

| Metric | Definition |
|---|---|
| WER | Corpus word error rate: (substitutions + deletions + insertions) / reference words |
| CER | Same at character level |
| Term recall | Share of `terms` found contiguously in the output |
| PII leak rate | Share of `pii` values whose digits appear anywhere in the output, whatever the spacing |
| Latency p50 / p95 | End-to-end per case, uncached calls only |
| Correction rejected | Cases where the guard discarded the LLM's proposal |

Protected values are masked out of both texts before WER is computed, so redacting (a governance choice) neither helps nor hurts accuracy; leaks are counted separately.

**Arms.** `wall-e` (any variant), `generic-whisper` (a stand-in for consumer dictation: plain Whisper, no priming, no correction, no redaction), `claude-pro` (imported by hand), and anything else you name.

## Capturing the Claude Pro arm

Claude Pro cannot be driven by a script, so its arm is captured once by a person following [PROTOCOL.md](../../../knowledge/stt/sv-medical/PROTOCOL.md): play each clip to the Claude app's voice input, copy the transcript it produced, put one JSON line per clip in `arms/claude-pro.jsonl`, and import it. The import is scored by exactly the same code as the Wall-E arm. Clips the app could not transcribe are recorded as empty and score as fully wrong, which is the honest outcome. Until that file exists, the loop's target falls back to the absolute WER in `loop.yaml`, and the report says so.

## Loop engineering

`npm run stt:loop` runs an optimisation loop over the pipeline (`server/src/services/stt/loop.ts`):

1. **Target.** The latest imported `claude-pro` run's objective (WER by default), or `target.wer` from `loop.yaml` when none exists, minus `margin`.
2. **Start.** Evaluate the `start` variant on the benchmark.
3. **Coordinate descent.** For each dimension in `search` (`stt_model`, `prompt`, `correction`, `correction_model`, `temperature`), try every alternative value from the incumbent. A candidate replaces the incumbent only when it lowers the objective *and* passes every guardrail (minimum term recall, maximum PII leak rate, p95 latency, correction rejection share, and a maximum share of cases that failed outright, since WER is computed over scored cases only).
4. **Learn.** Reference words the incumbent still gets wrong in at least `learn.min_occurrences` cases (numbers, stopwords and short tokens excluded) are appended to `glossary.learned.yaml`. They enter the Whisper prompt and the correction prompt. The incumbent is re-evaluated; if the learned terms did not help, they are rolled back.
5. **Stop** when the target is beaten by the margin, when `patience` passes bring no improvement, or at `max_iterations`.

Every evaluation is persisted as an `stt_runs` row with the loop id and iteration; `state.json` and `report.md` under `runs/loop-<id>/` are rewritten after each evaluation, so an interrupted loop is inspectable and the whole history is reproducible. The exit status is 0 only when the target was met.

Why coordinate descent and not a grid: each candidate costs real provider calls under free-tier rate limits (`budget.rpm`). The cache makes changing one knob cheap, so a pass over dimensions from a good incumbent explores what matters without paying for every combination. Add dimensions or values in `loop.yaml`; add a new knob to `SttVariant` and the loop picks it up.

## A local Swedish model (KB-Whisper) through Podman

KBLab's KB-Whisper models are Whisper fine-tuned on Swedish and are the obvious next step for this profile: better Swedish, and no audio leaving the host. Serve one with any OpenAI-compatible STT server and register it as a custom endpoint:

```bash
# example: an OpenAI-compatible faster-whisper server on the Podman host
podman run -d --name stt -p 8000:8000 \
  -e WHISPER__MODEL=KBLab/kb-whisper-large \
  ghcr.io/speaches-ai/speaches:latest        # or faster-whisper-server, whisper.cpp server, vLLM…

# register it in Wall-E (Keys page → custom endpoint, or the API)
curl -s http://localhost:3001/api/media/custom -H "Authorization: Bearer $DASHBOARD_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"baseUrl":"http://host.docker.internal:8000/v1","apiKey":"","modality":"transcription","modelId":"KBLab/kb-whisper-large","displayName":"KB-Whisper large"}'
```

Then put the model id in `loop.yaml` under `search.stt_model` and run the loop; it will adopt the local model if the benchmark says it is better. The server image, model format (CTranslate2 for faster-whisper) and flags depend on the server you pick; this was not exercised in this repository's tests.

## Governance for patient speech

- **Redaction** uses the knowledge governance policy (`knowledge/governance.yaml`). This work adds a `personnummer` detector (YYMMDD-NNNC and YYYYMMDD-NNNC, `-`, `+`, space or no separator; date plausibility incl. samordningsnummer; Luhn check digit) and turns it on. Spoken forms are normalised first so "85 04 12 12 30" is caught.
- **No transcript text is persisted.** `stt_transcripts` holds sha256 of the audio, the raw and the final text, model identities, prompt/glossary/policy hashes, redaction counts and latency. Benchmark rows in `stt_runs` hold scores, never references or hypotheses. The provider-output cache under `.cache/` does hold benchmark transcripts; it is git-ignored and only used for the synthetic benchmark.
- **Model policy** applies to the correction and structuring passes: `generation.allowed_models` / `denied_models` are enforced, and classifier models (gpt-oss-safeguard, prompt-guard) are now denied by default so the router never hands them a prompt.
- **Raw text** (before redaction) is only returned when the caller sets `include_raw=true`.
- **Data residency** is a deployment choice: with a local STT endpoint and a local correction model, no audio or text leaves the host. With Groq/Cloudflare, audio is sent to those providers under their terms. Nothing here makes an installation compliant with Swedish health-data law by itself; it gives the operator the controls and the audit trail to build a compliant deployment.

## API and CLI reference

**`POST /v1/stt/transcriptions`** (API key or client-profile key; multipart):

| Field | Default | Meaning |
|---|---|---|
| `file` | required | audio, ≤ 25 MB |
| `profile` | `sv-medical` | profile directory under `knowledge/stt/` |
| `model` | `auto` | STT model id, or `auto` for the router chain |
| `language` | profile language | ISO code passed to the recogniser |
| `prompt` | `glossary` | `glossary` primes Whisper with the profile vocabulary; `none` |
| `temperature` | `0` | recogniser temperature |
| `correction` | `llm` | `llm` or `off` |
| `correction_model` | router | pin `platform/model_id` |
| `redact` | `true` | apply PII redaction |
| `structure` | `false` | add journal JSON |
| `include_raw` | `false` | return `raw_text` and `corrected_text` too |

Response: `text`, `correction` (model, applied/rejected, distance), `redactions`, `structured`, `provenance` (id, STT model and duration, prompt/glossary/policy hashes), `latency_ms`; headers `X-Provider`, `X-Model`.

**Dashboard API** (session): `GET /api/stt/status`, `GET /api/stt/runs?profile=&arm=&loop=&limit=`, `GET /api/stt/runs/:id`, `GET /api/stt/glossary/:profile`, `GET /api/stt/transcripts`.

**CLI:** `npm run stt:synth`, `npm run stt:bench -- [--arm] [--model] [--prompt] [--correction] [--correction-model] [--no-redact] [--limit] [--tags] [--ids] [--rpm] [--no-cache] [--out] [--json] | --import <file> --arm <name> | --compare`, `npm run stt:loop -- [--config] [--iterations] [--limit] [--tags] [--no-cache]`.

## Limitations and honest caveats

- The benchmark audio is synthetic. Real consultations add noise, dialect, overlap and hesitation; expect higher error rates and re-run the loop on real recordings before trusting a configuration.
- The Claude Pro arm is captured by hand and depends on the app's dictation at the time of capture; record the date and settings with it. It cannot be re-run automatically.
- Free-tier rate limits shape the loop's speed: `budget.rpm` throttles recogniser calls and the correction pass waits out announced reset windows. A single Groq key is enough for the shipped benchmark but slow for many iterations.
- The correction pass is only as good as the free models the router reaches; the guard protects against rewrites, not against subtle wrong "corrections" of a rare term. Term recall in the benchmark is the check for that.
- Run-to-run variance: with `correctionModel` unset the router may hand each case to a different free model, so two evaluations of the same variant can differ by a percentage point of WER. The learning step re-evaluates the incumbent after every glossary change, which is correct, but part of the measured gain can be routing noise. Pin the correction model when its provider's rate limits allow it; the loop rejects a pinned model that runs out of keys through the error-rate guardrail.
- Names are not redacted (no NER); personnummer, phone, email and the other governance detectors are.
- No dashboard page yet for STT runs; the API exists and the CLI prints reports.
