# Tech Notes: Who Said What Brain

As of **15 Oct 2026**. This file merges four research notes (Gemma 4, FalkorDB, voice + deployment, event extras) with the verifier pass on their decision-critical claims.

**Conventions**
- **[unverified]**: not confirmed on an official page or by the verifier. Treat it as a hypothesis to test.
- **[inference]**: our own reading of the evidence.
- **[calc]**: our own arithmetic.
- **[team-measured]**: measured in the team's earlier projects under `D:\HACKATHONS\hackoctober fest\reference\` (read only).
- **[corrected]**: the verifier marked the original claim `partially_true`, and the text here is the corrected version. See the ledger in section 6.
- Section 5 (event extras) had **no verifier pass**. Its claims rest on one researcher's reading of the cited pages.

---

## 0. Decisions at a glance

1. **Gemma 4 hears audio only on E2B, E4B and 12B Unified.** Each clip is at most 30 s of 16 kHz mono float32 audio, and output is always text ([model card](https://ai.google.dev/gemma/docs/core/model_card_4), [audio guide](https://ai.google.dev/gemma/docs/capabilities/audio)). **No official serverless API we checked serves an audio-capable variant** [corrected: this absence cannot be proven exhaustively]. Hosted Gemma 4 is therefore text plus image (26B A4B or 31B). Spoken-question ASR needs a self-hosted E2B/E4B/12B or a separate ASR model.
2. **Hosted Gemma 4 text options:**
   - **Gemini API:** `gemma-4-31b-it` and `gemma-4-26b-a4b-it` are free of charge. The paid tier is "Not available", and free-tier data is used to improve Google products ([pricing](https://ai.google.dev/gemini-api/docs/pricing)).
   - **DigitalOcean Serverless Inference:** `gemma-4-31B-it` at **$0.18 / $0.50 per 1M input/output tokens**, prepaid ([DO pricing](https://docs.digitalocean.com/products/inference/details/pricing/)).
   - **Together AI no longer serves Gemma 4 serverless.** It removed it on 2026-09-15 [corrected] ([Together deprecations](https://docs.together.ai/docs/deprecations)).
3. **Diarization stays a separate component.** Nemotron-3-Diarization with multitalker Parakeet already runs live on this exact laptop [team-measured]: 182 ms per 1.12 s chunk and 3.2 GB peak VRAM with 4 ASR streams. **But MLH rule 9 and the 13–18 Oct build window mean we must not copy code from `reference\`.** Re-implement the pipeline inside the window on the same public models and libraries, and disclose this (section 5.4).
4. **FalkorDB: pin the engine version.**
   - `latest` (6.0.2, Rust engine) has **open, silent wrong-result bugs** in UNWIND/multi-MATCH query shapes ([#3343](https://github.com/FalkorDB/FalkorDB/issues/3343), [#2972](https://github.com/FalkorDB/FalkorDB/issues/2972)).
   - The C-engine tag is `falkordb/falkordb:v4.22.0`.
   - Override the Docker default `MAX_QUEUED_QUERIES 25 TIMEOUT 1000`.
   - Ingest in `UNWIND` micro-batches, because writes are serialized per graph.
5. **Live URL topology:**
   - Vercel Hobby for the UI and a stateless API. Limits: 300 s max duration, 4.5 MB body, WebSockets in beta.
   - FalkorDB on a **persistent** host. Do **not** use FalkorDB Cloud Free: it has no persistence, is stopped after 1 idle day and is deleted after 7 ([free tier](https://docs.falkordb.com/cloud/tiers/free)).
   - Pre-ingest demo meetings so the URL works without the laptop.
6. **Licences:**
   - Gemma 4 is Apache 2.0.
   - The FalkorDB server is SSPLv1, and the `main` branch LICENSE also appends the ELv2 text. SSPL is not OSI open source, so never call FalkorDB "open source" in the pitch.
   - Track 2 requires a public GitHub repo with an open-source licence ([Track 2 challenge](https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-a0/challenges/01a0d987-0c41-12ea-13ac-5b85755dc2f3)).
7. **Node 20 is too old for the npm `falkordb` 6.8.0 client**, which needs Node >=22 ([npm](https://registry.npmjs.org/falkordb/latest)). Use the Python client (`FalkorDB` 1.8.0) in the backend.

---

## 1. Gemma 4

### 1.1 Release timeline

| Date | Event | Source |
|---|---|---|
| 31 Mar 2026 | Release log: Gemma 4 in E2B, E4B, 31B and 26B A4B | https://ai.google.dev/gemma/docs/releases |
| **2 Apr 2026** | Public launch post. Cite this as the launch date. | https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ |
| 3 Apr 2026 | Vertex MaaS `gemma-4-26b-a4b-it-maas` released, stage "Experimental" | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it |
| 16 Apr 2026 | MTP (multi-token prediction) drafters for E2B, E4B, 31B and 26B A4B | https://ai.google.dev/gemma/docs/releases |
| 3 Jun 2026 | **Gemma 4 12B "Unified"** (encoder-free) | https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/ |
| ~5 Jun 2026 | QAT weights appear in Ollama (v0.30.6) | https://github.com/ollama/ollama/releases?page=5 |

### 1.2 Variants, modalities and audio support

Sources: https://ai.google.dev/gemma/docs/core/model_card_4 and the HF cards (https://huggingface.co/google/gemma-4-E2B-it, https://huggingface.co/google/gemma-4-E4B-it, https://huggingface.co/google/gemma-4-12B-it, https://huggingface.co/google/gemma-4-31b-it).

| Variant | HF id (instruction-tuned) | Params | Architecture | Context | Input | **Audio in** | Output | Encoders |
|---|---|---|---|---|---|---|---|---|
| **E2B** | `google/gemma-4-E2B-it` | 2.3B effective (5.1B with embeddings) | Dense, Per-Layer Embeddings, 35 layers | 128K | Text, image, **audio**, video (as frames) | **Yes** | Text | Vision ~150M, audio ~300M |
| **E4B** | `google/gemma-4-E4B-it` | 4.5B effective (8B with embeddings) | Dense, PLE, 42 layers | 128K | Text, image, **audio**, video (as frames) | **Yes** | Text | Vision ~150M, audio ~300M |
| **12B Unified** | `google/gemma-4-12B-it` | 11.95B | Dense, **encoder-free** (linear projections of image patches and raw audio), 48 layers | 256K | Text, image, **audio**, video (as frames) | **Yes** | Text | None |
| **26B A4B** | `google/gemma-4-26B-A4B-it` | 25.2B total / 3.8B active | MoE, 128 experts (8 active + 1 shared), 30 layers | 256K | Text, image, video (as frames) | No | Text | Vision ~550M |
| **31B** | `google/gemma-4-31B-it` | 30.7B | Dense, 60 layers | 256K | Text, image, video (as frames) | No | Text | Vision ~550M |

- **Output is text only on every variant.** TTS must come from another component ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
- **Video** is processed as frames, up to 60 s at 1 fps.
- **Image visual-token budgets** are 70, 140, 280, 560 and 1120.
- **Training-data cutoff** is January 2025 ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
- **Also in the HF collection:** the `*-assistant` MTP drafters and `google/diffusiongemma-26B-A4B-it` ([collection](https://huggingface.co/collections/google/gemma-4)).

### 1.3 Audio input details

- **Which variants:** native audio exists on E2B, E4B and 12B Unified only. At the April launch only E2B and E4B had audio ([launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)). The 12B is the first medium-sized model that ingests audio natively ([12B dev guide](https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide/)).
- **Tasks:** ASR, speech-to-translated-text (AST), and audio understanding (Q&A or a summary over one or more clips; the docs summarise five files) ([audio guide](https://ai.google.dev/gemma/docs/capabilities/audio)).
- **Limits:**
  - **At most 30 s per clip.**
  - **16 kHz, mono, float32 in [-1, 1].**
  - **25 tokens per second of audio** (Gemma 3n used 6.25).
  - **No streaming or real-time mode is documented.** The mic example records a whole clip first ([audio guide](https://ai.google.dev/gemma/docs/capabilities/audio)).
  - [calc] A 30 s clip is about 750 tokens. An hour of chunked audio is about 90K tokens.
- **Prompting:**
  - Put the audio after the text.
  - The placeholder is `<|audio|>`, wrapped as `<|audio>`…`<audio|>`.
  - Recommended ASR prompt: "Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text…", with digits for numbers and no newlines.
  - Sources: [prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4), [audio guide](https://ai.google.dev/gemma/docs/capabilities/audio).
- **12B mechanism:** raw 16 kHz audio is sliced into 40 ms frames (640 floats each) and projected linearly into the LLM. E2B and E4B use a 12-layer conformer encoder ([12B dev guide](https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide/)).
- **Audio benchmarks** ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)):
  - CoVoST: 12B 38.5\*, E4B 35.54, E2B 33.47.
  - FLEURS (lower is better): 12B 0.069\*, E4B 0.08, E2B 0.09.
  - \*12B figures exclude Chinese.
- **Diarization:**
  - The 12B developer guide lists "diarization" among the capabilities, with no benchmark or example ([12B dev guide](https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide/)).
  - The model card and the audio docs never mention diarization.
  - **Gemma diarization quality is [unverified].** Keep a dedicated diarizer.
- **Inconsistency:** the 12B guide demos a 5-minute video (313 frames) with its audio track. That conflicts with the 30 s audio and 60 s video limits. Audio longer than 30 s on 12B is **[unverified]** ([12B dev guide](https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide/)).

### 1.4 Licence

- **Apache 2.0**, which the launch post calls "commercially permissive". Gemma 1 to 3n used the Gemma Terms of Use instead. Sources: [launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [model card](https://ai.google.dev/gemma/docs/core/model_card_4), and the HF metadata `apache-2.0` ([E4B card](https://huggingface.co/google/gemma-4-E4B-it)).
- The Gemma 4 licence page is the unmodified Apache 2.0 text ([apache_2](https://ai.google.dev/gemma/apache_2)).
- The Gemma Terms of Use cover only Gemma 1 to 3n ([terms](https://ai.google.dev/gemma/terms)).
- Whether the Gemma Prohibited Use Policy also binds Apache-licensed Gemma 4 is **[unverified]** ([policy](https://ai.google.dev/gemma/prohibited_use_policy)).

### 1.5 Function calling, structured output, thinking

- **Native function calling:**
  - Six tool tokens: `<|tool>`/`<tool|>`, `<|tool_call>`/`<tool_call|>`, `<|tool_response>`/`<tool_response|>`.
  - String values are delimited by `<|"|>`.
  - In Transformers, pass `tools=` to `processor.apply_chat_template(...)`. Tools can be JSON schemas, or Python functions converted with `get_json_schema()`.
  - This works on E2B, E4B, 12B, 26B-A4B and 31B. The model never executes code itself, so use an allowlist.
  - Sources: [prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4), [function calling](https://ai.google.dev/gemma/docs/capabilities/function-calling).
- **Structured output:**
  - The launch blog claims "structured JSON output" ([blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)), but the prompt docs define no dedicated JSON mode.
  - Vertex MaaS lists structured output as supported ([MaaS](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it)).
  - Ollama v0.21.1 fixed structured outputs for Gemma 4 with think=false ([releases p6](https://github.com/ollama/ollama/releases?page=6)).
- **Thinking:**
  - Turn it on with `<|think|>` at the start of the system prompt, or with `enable_thinking=True`.
  - Output appears in `<|channel>thought` … `<channel|>`. Parse it with `processor.parse_response()`, decoding with `skip_special_tokens=False`.
  - With thinking off, 12B, 26B and 31B still emit an empty thought block.
  - Strip earlier thoughts from multi-turn history ([thinking](https://ai.google.dev/gemma/docs/capabilities/thinking)).
  - In the Gemini API, set `thinking_level` to "high" or "minimal" ([Gemma on Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)).
- **Other:**
  - Native `system` role.
  - Recommended sampling: temperature 1.0, top_p 0.95, top_k 64.
  - MTP drafters for speculative decoding.
  - Source: [model card](https://ai.google.dev/gemma/docs/core/model_card_4).

### 1.6 Model IDs per platform

| Platform | Gemma 4 identifiers | Audio on this platform | Source |
|---|---|---|---|
| Hugging Face weights | `google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}` and `…-it`. MTP drafters: `…-it-assistant`. QAT: `…-it-qat-q4_0-gguf`, `…-it-qat-q4_0-unquantized`, `…-it-qat-w4a16-ct`. Mobile: `google/gemma-4-{E2B,E4B}-it-qat-mobile-ct` / `-qat-mobile-transformers` | **Yes** for E2B, E4B and 12B via Transformers (`AutoProcessor` + `AutoModelForMultimodalLM`; audio item `{"type":"audio","audio":<url\|path>}`; `pipeline(task="any-to-any")`; needs `librosa`). Minimum transformers version [unverified]. | https://huggingface.co/collections/google/gemma-4 , https://huggingface.co/google/gemma-4-E4B-it , https://ai.google.dev/gemma/docs/capabilities/audio |
| HF Inference Providers | `google/gemma-4-31B-it` (Novita, Featherless AI, DeepInfra live; Together in error status). `google/gemma-4-26B-A4B-it` (Novita, Featherless AI, Scaleway, DeepInfra) | **No.** E2B-it, E4B-it and 12B-it have no providers. | https://huggingface.co/api/models/google/gemma-4-26B-A4B-it?expand[]=inferenceProviderMapping , https://huggingface.co/google/gemma-4-12B-it |
| Ollama | `gemma4` / `gemma4:latest` (= e4b), `gemma4:e2b`, `gemma4:e4b`, `gemma4:12b`, `gemma4:26b` (alias `gemma4:26b-a4b`), `gemma4:31b`. Cloud: `gemma4:cloud`, `gemma4:31b-cloud`. QAT: `gemma4:e2b-it-qat` (4.3GB), `gemma4:e4b-it-qat` (6.1GB), `gemma4:12b-it-qat` (7.2GB), `gemma4:26b-a4b-it-qat` (16GB), `gemma4:31b-it-qat` (19GB). Also `-it-q4_K_M`, `-it-q8_0`, `-it-bf16`, `-mxfp8`, `-nvfp4`, `-mlx`, `-mlx-bf16` and MTP tags. Official GGUF pull: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0` | **Flaky and version-dependent on Windows/CUDA** [corrected]. See below. | https://ollama.com/library/gemma4/tags , https://ollama.com/library/gemma4 , https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf |
| llama.cpp | `-hf ggml-org/gemma-4-E2B-it-GGUF`, `-hf ggml-org/gemma-4-E4B-it-GGUF` (documented audio + vision). `ggml-org/gemma-4-12B-it-GGUF` has an audio projector but is undocumented. | E2B/E4B **documented**. 12B present upstream, **end-to-end [unverified]** [corrected] | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md , https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/main/convert.log |
| Google official GGUF | `google/gemma-4-E4B-it-qat-q4_0-gguf`: `gemma-4-E4B_q4_0-it.gguf` (5.15 GB) + `gemma-4-E4B-it-mmproj.gguf` (0.99 GB). 12B QAT Q4_0 GGUF is 6.98 GB. | Whether Google's mmproj enables audio in llama.cpp is [unverified] | https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main , https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf |
| Gemini API / AI Studio | `gemma-4-31b-it`, `gemma-4-26b-a4b-it` (google-genai `client.models.generate_content(...)`, REST `v1beta/models/gemma-4-31b-it:generateContent`). Supports thinking, image input, system instructions, function calling and Google Search. | **No.** The page header mentions audio, but the served variants have no audio and there is no audio example. | https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api |
| Vertex AI MaaS | `gemma-4-26b-a4b-it-maas` only. Experimental, region `global`, context 262,144, max output 128,000. Function calling and structured output yes; **thinking not supported**. Prices not on the page. | **No** (audio and video "not supported") | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/maas/google/gemma-4-26b-a4b-it |
| Vertex Model Garden (self-deploy) | slug `gemma4`: 31B, 26B A4B, E4B, E2B (12B not listed) | E4B/E2B list "Text, image and audio". Whether the serving container exposes audio is [unverified]. | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma |
| Cloudflare Workers AI | `@cf/google/gemma-4-26b-a4b-it` (256K context, function calling, reasoning, vision). $0.10 / $0.30 per M in/out; 10,000 free Neurons/day. | No | https://developers.cloudflare.com/ai/models/@cf/google/gemma-4-26b-a4b-it/ , https://developers.cloudflare.com/workers-ai/platform/pricing/ |
| OpenRouter | `google/gemma-4-31b-it`, `google/gemma-4-26b-a4b-it` (accept `tools`, `response_format`, `reasoning`). Price and `:free` variant [unverified]. | No. 26B input is image, text and video; 31B modalities [unverified] [corrected] | https://openrouter.ai/google , https://openrouter.ai/google/gemma-4-31b-it/llms.txt |
| DigitalOcean Serverless Inference | `gemma-4-31B-it` (256K context, 8,192 max output, Chat Completions + Responses). **$0.18 / $0.50 per 1M in/out**, cache read $0.036. Prepaid only. | No | https://docs.digitalocean.com/products/inference/details/models/ , https://docs.digitalocean.com/products/inference/details/pricing/ , https://docs.digitalocean.com/products/inference-hub/details/models/ |
| Together AI | **None serverless as of 15 Oct 2026.** `google/gemma-4-31B-it` was removed from serverless on 2026-09-15 (replacement `zai-org/GLM-5.3-Flash`) [corrected] | n/a | https://docs.together.ai/docs/deprecations |
| Groq | None listed | n/a | https://console.groq.com/docs/models |
| vLLM (self-host) | `google/gemma-4-E4B-it` with `--limit-mm-per-prompt '{"image": 4, "audio": 1}'`. "Fits on a single 24 GB+ GPU". | **Yes** (OpenAI `audio_url`) | https://recipes.vllm.ai/Google/gemma-4-E4B-it |
| LiteRT-LM | `litert-community/gemma-4-{E2B,E4B,12B}-it-litert-lm`. CLI: `litert-lm run <model> --audio-backend=gpu --attachment=audio.wav --prompt="Transcribe this audio."` | CLI yes (Windows + GPU). Audio via `serve` mode is [unverified]. | https://developers.google.com/edge/litert-lm/cli/usage |

**Ollama audio, corrected picture:**
- **Official signals:**
  - The library page lists modalities **Text, Image, Audio for E2B and E4B**, a ~300M audio encoder and an `audio` badge ([library](https://ollama.com/library/gemma4)).
  - On `/tags`, every tag's Input column says only "Text, Image" ([tags](https://ollama.com/library/gemma4/tags)).
  - Release v0.20.0-rc1 reads "convert: support new Gemma4 audio_tower tensor naming (#15221)".
  - Release v0.33.3 reads "gemma4 now supports images and audio on MLX engine" ([v0.33.3](https://github.com/ollama/ollama/releases/tag/v0.33.3)).
  - Ollama's API docs show no audio field in `/api/chat` and no `/v1/audio/transcriptions` ([api/chat](https://docs.ollama.com/api/chat), [openai-compat](https://docs.ollama.com/api/openai-compatibility)). The exact request shape for audio is therefore [unverified].
- **User reports on Windows/NVIDIA (not official):**
  - Issue #16584 (Windows 11, RTX 3090) says `gemma4:e4b` transcription worked on v0.20.7 and regressed on v0.30.6, where thinking is on by default. 12b works with thinking disabled (https://github.com/ollama/ollama/issues/16584).
  - Issue #15333 (v0.20.2, Windows) shows audio encoding working, but the forward pass hits intermittent GGML crashes (https://github.com/ollama/ollama/issues/15333).
- **Verdict:** flaky and version-dependent. Test the exact Ollama version before relying on it.

**llama.cpp audio, corrected picture:**
- `docs/multimodal.md` lists Gemma 4 audio + vision only for the E2B and E4B GGUFs. 26B-A4B and 31B are vision only, and 12B is not listed ([multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)).
- The "experimental" multimodal label is in `tools/server/README.md`. The same README documents OpenAI `input_audio` parts (data or url; mp3/wav/flac decoded by miniaudio) on `/v1/chat/completions` ([server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).
- 12B audio support exists in code:
  - `ggml-org/gemma-4-12B-it-GGUF`'s mmproj sets `clip.audio.projector_type` to `gemma4ua` ([convert.log](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/main/convert.log)).
  - PR #24088 (merged 3 Jun 2026) fixed mtmd for the encoder-free 12B.
  - Issue #24084 was closed by #24091.
- End-to-end 12B audio is [unverified].

### 1.7 Local fit on the RTX 4060 Laptop (8 GB VRAM, 48 GB RAM)

**Official weight memory.** These figures include 20% overhead but **exclude KV cache and runtime** ([Gemma core docs](https://ai.google.dev/gemma/docs/core)):

| Model | BF16 | SFP8 | Q4_0 |
|---|---|---|---|
| E2B | 11.4 GB | 5.7 GB | 2.9 GB |
| E4B | 17.9 GB | 8.9 GB | 4.5 GB |
| 12B | 26.7 GB | 13.4 GB | 6.7 GB |
| 26B A4B | 57.7 GB | 28.8 GB | 14.4 GB (all params must load) |
| 31B | 69.9 GB | 34.9 GB | 17.5 GB |

**Fit by variant** ([inference] unless cited):
- **E2B or E4B at Q4 fits fully**, with room for KV cache. These are the safe local audio models.
  - Ollama `gemma4:e2b-it-qat` is 4.3 GB and `gemma4:e4b-it-qat` is 6.1 GB ([tags](https://ollama.com/library/gemma4/tags)).
  - The E4B GGUF is 5.15 GB plus a 0.99 GB mmproj.
- **E4B at Q8 or SFP8** (8.9 GB) does not fit and needs partial CPU offload. **BF16 Transformers** (11.4 GB / 17.9 GB) does not fit either.
  - Whether `-qat-w4a16-ct` or `-qat-mobile-transformers` loads in plain Transformers on Windows is [unverified].
- **12B at Q4 is borderline.**
  - Sizes: 6.7 GB official, 7.2 GB in Ollama, 6.98 GB as GGUF.
  - Google targets "16GB VRAM or unified memory" laptops ([12B blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/)).
  - LiteRT-LM measured about 7.3–7.9 GB of GPU memory for the 6883 MB model ([litert 12B](https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm)).
- **26B A4B and 31B** need heavy CPU offload; 48 GB RAM is enough. Their speed is [unverified], and they take no audio.

**Published throughput.** These are LiteRT-LM decode figures (text only; 1024 prefill / 256 decode tokens, 4096 context) from Google's litert-community cards [corrected: the 12B figures come from the HF card, not the 12B blog, which has no speed figures]:

| Model | Hardware (OS) | Prefill tok/s | Decode tok/s | TTFT | Source |
|---|---|---|---|---|---|
| E2B | RTX 4090 (Linux) | 11,234 | 143.4 | | https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm |
| E4B | RTX 4090 (Linux) | 7,260 | 91.2 | | https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm |
| 12B | RTX 4090 (Linux) | 3,548 | 69 | 0.3 s | https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm |
| 12B | RTX 5080 16 GB (Windows) | 391 | 50 | 2.5 s | same |
| 12B | MacBook M4 Pro GPU / MacBook Air M4 GPU / Web (M4 Max) | | 29 / 15 / 26 | | same |

- **No RTX 4060 figures exist.** Benchmark locally with `llama-bench` or a timed `ollama run`.
- NVIDIA's Gemma 4 post publishes no tokens/sec figures ([NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-open-models)).

### 1.8 Hosted options: summary

- **Text and image Q&A, no GPU of our own** (choose by track; see 4.3):
  - Gemini API (free; data used for product improvement).
  - DigitalOcean `gemma-4-31B-it` (paid, prepaid).
  - Cloudflare `@cf/google/gemma-4-26b-a4b-it`.
  - OpenRouter.
- **Audio-capable Gemma 4 in the cloud means self-hosting:**
  - vLLM on a GPU.
  - Cloud Run GPU or GKE ([Google Cloud integration](https://ai.google.dev/gemma/docs/integrations/google-cloud)).
  - Vertex Model Garden self-deploy.
  - An HF ZeroGPU Space (section 4.1).
  - The 12B guide also describes the local OpenAI-compatible `litert-lm serve` and self-deploy via Model Garden, Cloud Run or GKE, but **no hosted serverless API** [corrected] ([12B dev guide](https://developers.googleblog.com/en/gemma-4-12b-the-developer-guide/)).
- **Not fully proven absent:** a third-party aggregator (Eden AI) lists `deepinfra/google/gemma-4-E4B-it` with "Audio input", but deepinfra.com/google/gemma-4-E4B-it returns 404 [unverified].

### 1.9 Official voice and audio examples

- **Audio guide and notebook** ([guide](https://ai.google.dev/gemma/docs/capabilities/audio), [notebook](https://github.com/google-gemma/cookbook/blob/main/docs/capabilities/audio.ipynb)):
  - Covers ASR, AST and multi-clip understanding on E2B, E4B and 12B, and runs on a T4.
  - Mic capture uses `ipywebrtc` `AudioRecorder` with webm → wav via ffmpeg. This records a clip; it does not stream.
- **Sample audio:** `https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav` ([E4B card](https://huggingface.co/google/gemma-4-E4B-it)).
- **No official real-time or voice-agent example exists for Gemma 4.** The low-latency "Live API" is Gemini-only ([cookbook tree](https://api.github.com/repos/google-gemma/cookbook/git/trees/main?recursive=1)).
- [inference] A live loop is VAD/chunking (≤30 s) → ASR → text Q&A → separate TTS.

---

## 2. FalkorDB

### 2.1 Version pinning and the open Rust-engine bugs

**Releases** ([tags](https://github.com/FalkorDB/FalkorDB/tags), [releases](https://github.com/FalkorDB/FalkorDB/releases)):

| Version | Date | Notes |
|---|---|---|
| **6.0.2** (Latest) | 8 Oct 2026 | Rust engine |
| 6.0.1 | 1 Oct 2026 | |
| 4.22.0 | 30 Sep 2026 | C engine. Docker Hub tag `v4.22.0`, amd64/arm64. |
| 6.0.0 | 25 Sep (tag) / 29 Sep (release) | First Rust-engine release |
| 4.20.7 | 24 Sep 2026 | |

**What 6.0.0 changed** ([v6.0.0](https://github.com/FalkorDB/FalkorDB/releases/tag/v6.0.0)):
- The engine is rewritten in Rust. Commands, Cypher and clients are unchanged.
- Built on Redis 8.10.2.
- Full-text and vector indexes run on RediSearch 8.6.1.
- `latest` points to the Rust engine. `latest-alpine` and the `v4.x.y` / `edge-c` tags stay on the C engine.
- 4.x and 6.0 nodes cannot replicate with each other.
- Not yet in 6.0: `algo.AStar`, CCH, Leiden/Louvain.

**What 6.0.2 fixed** ([v6.0.2](https://github.com/FalkorDB/FalkorDB/releases/tag/v6.0.2)):
- A wrong-dimension vector no longer removes the entity from every index on its label (#3087).
- KNN `k` is clamped (#3088).
- Vector OPTIONS validation now matches 4.x (#3094).
- The `redis-server` runs as non-root unless `SKIP_DROP_PRIVS=1`.
- **None of these fixes the bugs below.**

**Open silent wrong-result bugs:**
- **#3343** (opened 14 Oct 2026, open, no labels) [corrected] ([#3343](https://github.com/FalkorDB/FalkorDB/issues/3343)):
  - `UNWIND … MATCH … WHERE … MATCH` silently returns 0 rows on `6.0.2` and `:edge`. EXPLAIN shows no Unwind op.
  - The issue lists **three workarounds**:
    - (a) put `WITH n` between the MATCH clauses;
    - (b) use an inline map, `MATCH (n:Fact {id: x})`;
    - (c) use `WHERE id(n) IN $ids` without UNWIND.
  - The issue never mentions the C engine. **Whether 4.22.0 is free of this bug is [unverified].**
- **#2972** (open, labelled rust) ([#2972](https://github.com/FalkorDB/FalkorDB/issues/2972)):
  - `MATCH … WHERE` followed by a traversing `MATCH` drops the WHERE, and a following `DETACH DELETE` deletes unmatched data.
  - The C engine behaves correctly for this case.
  - The team hit the same WHERE drop on 6.0.1 [team-measured] (`reference\FalkorDB\docs\falkordb-issues.md`, issue 3).

**Choice:**
- (1) Pin `falkordb/falkordb:v4.22.0` (C engine; 4.x tags carry a `v` prefix, 6.x tags do not) ([Docker Hub tags](https://hub.docker.com/v2/repositories/falkordb/falkordb/tags?page_size=40)).
- (2) Or pin `:6.0.2` and enforce the query rules: a `WITH` barrier between consecutive MATCH clauses, and inline property maps after UNWIND.
- **Never use bare `:latest`.**

**Speed on this laptop.** On Windows 11 + Docker Desktop/WSL2 with a 4-process client, 6.0.1 beat 4.22.0 [team-measured] (`reference\FalkorDB\docs\bench\README.md`):
- Throughput: ~1.8k vs ~1.36k ops/s.
- Read p50: 8.5 ms vs 11 ms.

### 2.2 Cypher support

FalkorDB implements a subset of openCypher (v9 reference) plus extensions ([cypher-support](https://docs.falkordb.com/cypher/cypher-support.html)).

**Supported:**
- `MATCH`, `OPTIONAL MATCH`, `WHERE`, `ORDER BY`, `SKIP`, `LIMIT`, `RETURN`, `WITH`, `UNWIND`
- `CREATE`, `MERGE` (with `ON CREATE SET` / `ON MATCH SET`), `SET`, `REMOVE`, `DELETE`, `DETACH DELETE`
- `UNION` / `UNION ALL`
- `CALL` procedures and `CALL {}` subqueries ([call](https://docs.falkordb.com/cypher/call.md))
- `FOREACH` and `LOAD CSV` ([docs index](https://docs.falkordb.com/_llms/falkor-db/documentation.txt))
- Parameters and temporal types

**Gaps compared with Neo4j:**
- No regex operator. Use `string.matchRegEx` or the FLEX UDFs ([functions](https://docs.falkordb.com/cypher/functions.html)).
- No label expressions, temporal arithmetic, or hex/octal literals.
- No `exists((n)-[]->())` pattern checks.
- No schema queries. Use `db.labels()` and related procedures ([procedures](https://docs.falkordb.com/cypher/procedures.md)).
- No APOC ([neo4j migration](https://docs.falkordb.com/operations/migration/neo4j-to-falkordb.md)).
- No multi-query transactions. MULTI/EXEC blocks other graphs ([concurrency](https://docs.falkordb.com/design/concurrency.md)).
- `CALL {} IN TRANSACTIONS` is [unverified].
- Bolt is experimental and "not meant to be used in production" (`BOLT_PORT 7687`; only the Python `neo4j` driver is tested) ([bolt](https://docs.falkordb.com/integration/bolt-support.md)).

**Known limitations** ([known-limitations](https://docs.falkordb.com/cypher/known-limitations.html)):
- LIMIT does not constrain eager writes.
- `<>` filters cannot use indexes.
- An unreferenced relationship alias only checks that one matching edge exists.
- Aggregates are not allowed in pattern comprehensions.

**Team learning:** FOREACH writes run after the whole UNWIND batch has matched [team-measured] (`reference\FalkorDB\cypher\10_ingest_episode.cypher`).

**Extensions:**
- JavaScript UDFs (`GRAPH.UDF LOAD`; falkordb-py `udf_load`, `udf_list`, `udf_flush`, `udf_delete`) ([falkordb.py](https://raw.githubusercontent.com/FalkorDB/falkordb-py/main/falkordb/falkordb.py)).
- FLEX (`text.levenshtein`, `text.jaroWinkler`, `date.*`, `json.*`).
- Algorithms: `algo.pageRank`, `algo.BFS`, `algo.WCC`, `algo.betweenness`, `algo.labelPropagation`, `algo.SPpaths`, `algo.SSpaths`, `algo.MSF` ([procedures](https://docs.falkordb.com/cypher/procedures.md)).

### 2.3 Indexes: exact syntax

**Vector (HNSW)** ([vector-index](https://docs.falkordb.com/cypher/indexing/vector-index.md)):

```cypher
CREATE VECTOR INDEX FOR (u:Utterance) ON (u.embedding)
  OPTIONS {dimension: 768, similarityFunction: 'cosine'}
-- optional, Cypher-map syntax with colons: M:32, efConstruction:200, efRuntime:10
CREATE VECTOR INDEX FOR ()-[e:Call]->() ON (e.summary) OPTIONS {dimension:128, similarityFunction:'euclidean'}
CREATE (p:Product {description: vecf32([2.1, 0.82, 1.3])})
CALL db.idx.vector.queryNodes('Utterance', 'embedding', 10, vecf32($q)) YIELD node, score
CALL db.idx.vector.queryRelationships(relType, attr, k, vec) YIELD relationship, score
DROP VECTOR INDEX FOR (p:Product) ON (p.description)
```

- **Options:**
  - `dimension` and `similarityFunction` (`'euclidean'` or `'cosine'`) are **required**.
  - `M`, `efConstruction` and `efRuntime` are optional, with defaults 16 / 200 / 10. They are written as map entries with colons, not `=` [corrected].
  - Per-query efRuntime syntax is [unverified].
- **Dimensions:** the page says 1–4096, but its FAQ says "any dimension". The docs are inconsistent.
- **Score direction** [corrected]:
  - The docs are internally inconsistent. They call it "a similarity score" and every example uses `ORDER BY score DESC`, yet euclidean is described as "0 = identical".
  - The team measured `score` as a **distance (lower = closer)** on 6.0.1 [team-measured] (`reference\FalkorDB\cypher\00_schema.cypher`, line 2). The verifier did not reproduce this, so **re-check it on the pinned version** before choosing ASC or DESC.
- **Results are approximate.**
- **Functions:** `vecf32`, `vec.euclideanDistance`, `vec.cosineDistance` ([functions](https://docs.falkordb.com/cypher/functions.html)). `vecf64` is [unverified].
- **Live writes:** whether new vectors are indexed immediately is [unverified]. Check the `status` column of `CALL db.indexes()`.
- **Memory:** 3000 × 768-d vectors added ~12.6 MB `used_memory`, which `GRAPH.MEMORY USAGE` did not count [team-measured] (`reference\FalkorDB\docs\falkordb-issues.md`, issue 1).
- **falkordb-py helper:** `create_node_vector_index(label, *props, dim=0, similarity_function="euclidean")` ([graph.py](https://raw.githubusercontent.com/FalkorDB/falkordb-py/main/falkordb/graph.py)).

**Full-text (RediSearch)** ([fulltext-index](https://docs.falkordb.com/cypher/indexing/fulltext-index.html)):

```cypher
CREATE FULLTEXT INDEX FOR (u:Utterance) ON (u.text)
CREATE FULLTEXT INDEX FOR (n:Movie) ON (n.title) OPTIONS {language:'German', stopwords:['a','ab']}
CREATE FULLTEXT INDEX FOR ()-[m:Manager]-() ON (m.name)
CALL db.idx.fulltext.createNodeIndex({label:'Movie', language:'German'}, 'title')
CALL db.idx.fulltext.createNodeIndex('Movie', {field:'title', phonetic:'dm:en'})
CALL db.idx.fulltext.queryNodes('Utterance', $q) YIELD node, score        -- TF-IDF score
CALL db.idx.fulltext.queryRelationships('Manager', 'Charlie Munger') YIELD relationship
CALL db.idx.fulltext.drop('Movie')
DROP FULLTEXT INDEX FOR (n:Movie) ON (n.title)
```

- **Query syntax** is RediSearch: prefix `Jun*`, fuzzy `%term%1`, OR `|`, NOT `-`.
- Since 6.0.2, unknown option keys are rejected ([v6.0.2](https://github.com/FalkorDB/FalkorDB/releases/tag/v6.0.2)).
- `weight` and `nostem` syntax is [unverified].
- **Sanitize user text** before querying: a bare `_` broke the tokenizer [team-measured] (`reference\FalkorDB\cypher\00_schema.cypher`).

**Range (exact match and range)** ([range-index](https://docs.falkordb.com/cypher/indexing/range-index.html)):

```cypher
CREATE INDEX FOR (u:Utterance) ON (u.t_start)          -- legacy: CREATE INDEX ON :Person(age)
CREATE INDEX FOR ()-[f:FOLLOW]-() ON (f.created_at)
DROP INDEX ON :Person(age)
```

- Used for equality, `<`/`>`/`<=`/`>=`, `IN` on arrays, and geo `distance() <`.
- Indexable types: strings, numbers, points and scalar arrays. Maps, nested arrays and vectors are not indexable.
- The range page shows one property per index. The constraint page shows the composite form `CREATE INDEX FOR (p:Person) ON (p.first_name, p.last_name)` ([constraint-create](https://docs.falkordb.com/commands/graph.constraint-create.html)).
- Verify index use with `GRAPH.EXPLAIN` or `GRAPH.PROFILE`.

**Constraints** are a Redis command, not Cypher ([constraint-create](https://docs.falkordb.com/commands/graph.constraint-create.html)):
- Syntax: `GRAPH.CONSTRAINT CREATE <graph> UNIQUE|MANDATORY NODE <label>|RELATIONSHIP <type> PROPERTIES <n> <prop…>`.
- UNIQUE needs an exact-match index on the same properties first.
- Creation is asynchronous: `PENDING` → `UNDER CONSTRUCTION` → `OPERATIONAL`, or `FAILED`. Check with `CALL db.constraints()`.
- Cypher `CREATE CONSTRAINT` is [unverified].

### 2.4 Client SDKs

**Python: `FalkorDB` 1.8.0** (uploaded 2026-10-14; Python >=3.10; `redis>=8.0.0,<9`; optional `[hiredis]`) ([PyPI](https://pypi.org/pypi/FalkorDB/json)). **Recommended for this project.**
- **Sync:** `from falkordb import FalkorDB`; `FalkorDB(host, port, username=, password=, ssl=)`.
- **`from_url`:** `falkor://`, `falkors://` (TLS), `redis://`, `rediss://`, `unix://` ([falkordb.py](https://raw.githubusercontent.com/FalkorDB/falkordb-py/main/falkordb/falkordb.py)).
- **Async:** `from falkordb.asyncio import FalkorDB`, with `from_url`, `select_graph`, `aclose()` and `async with` ([asyncio](https://raw.githubusercontent.com/FalkorDB/falkordb-py/main/falkordb/asyncio/falkordb.py)). The README passes a `redis.asyncio.BlockingConnectionPool` as `connection_pool=` ([README](https://github.com/FalkorDB/falkordb-py)).
- **Graph API:** `query(q, params=None, timeout=None)`, `ro_query`, `explain`, `profile`, `copy`, `delete`.
- **Helpers:** `create_node_{range,fulltext,vector}_index`, `create_node_unique_constraint`, `create_node_mandatory_constraint`, `list_indices` ([graph.py](https://raw.githubusercontent.com/FalkorDB/falkordb-py/main/falkordb/graph.py)).

**JS/TS: npm `falkordb` 6.8.0** (repo `FalkorDB/falkordb-ts`, published 2026-08-25, **`engines.node >=22`**, built on `@redis/client` ^5.10) ([npm](https://registry.npmjs.org/falkordb/latest), [repo](https://github.com/FalkorDB/falkordb-ts)).
- Connect with `await FalkorDB.connect({host, port, username, password})`, `{url:'falkor://localhost:6379'}` or `{socket:{host, port, tls:true}}`.
- Pass parameters with `graph.query(cypher, {params:{...}})`.
- **The local Node 20 is too old.**

**Other clients:** jfalkordb, falkordb-go, falkordb-rs, NFalkorDB ([clients](https://docs.falkordb.com/getting-started/clients.md)).

**FalkorDBLite** (`pip install falkordblite`, Python >=3.12) is for development and testing. Windows support is [unverified] ([falkordblite](https://docs.falkordb.com/operations/falkordblite/falkordblite-py.md)).

### 2.5 GraphRAG-SDK and text-to-Cypher

**GraphRAG-SDK 1.4.0** (uploaded 2026-08-10; Python >=3.10; depends on `falkordb>=1.0,<2`, `gliner`, `hnswlib`, `tiktoken`; extras `litellm`, `openrouter`, `openai`, `anthropic`, `cohere`, `pdf`, `fastcoref`, `sentence-transformers`, `all`) ([PyPI](https://pypi.org/pypi/graphrag-sdk/json)).
- **API** ([README](https://raw.githubusercontent.com/FalkorDB/GraphRAG-SDK/main/README.md)):
  - Fully async: `async with GraphRAG(connection=ConnectionConfig(...), llm=LiteLLM(...), embedder=LiteLLMEmbedder(...)) as rag:`.
  - Workflow: `ingest(text=..., document_id=...)` → `finalize()` → `completion(question)`.
  - `completion(..., return_context=True)` returns the retrieval trail.
  - Incremental updates: `update`, `delete_document`, `apply_changes`. Call `finalize()` once per batch, because its cost scales with graph size.
  - Legacy API: `graphrag-sdk==0.8.2`.
- **`ConnectionConfig`:**
  - Fields: `host`, `port`, `username`, `password`, `graph_name` (default `knowledge_graph`).
  - Also `ConnectionConfig.from_url("redis://user:pass@host:port", graph_name=...)`.
  - The SDK creates its own vector indexes on `Chunk.embedding`, `__Entity__.embedding` and `RELATES.embedding` ([configuration](https://docs.falkordb.com/graphrag/configuration.md)).
- **Providers** ([providers](https://docs.falkordb.com/graphrag/providers.md)):
  - `LiteLLM`/`LiteLLMEmbedder` (default), `OpenRouterLLM`/`OpenRouterEmbedder`, or custom `LLMInterface` (implement `invoke`) and `Embedder` (implement `model_name` and `embed_query`).
  - **No Ollama or Gemini examples.** For local models the docs say to write a custom `LLMInterface`.
- **Gemma route:**
  - LiteLLM routes `ollama_chat/<model>` with `api_base="http://localhost:11434"` ([LiteLLM Ollama](https://docs.litellm.ai/docs/providers/ollama)) and `gemini/<model>` with `GEMINI_API_KEY` ([LiteLLM Gemini](https://docs.litellm.ai/docs/providers/gemini)).
  - Both are plausible for Gemma but **[unverified] end to end**.
  - Whether Gemma IDs are served under `gemini/` is [unverified].
  - For embeddings, use a custom `Embedder`; the team used EmbeddingGemma before (`reference\FalkorDB\docker-compose.yml`).
- **Extraction:** GLiNER NER (`urchade/gliner_medium-v2.1`), then an LLM call per chunk. Speaker and timestamp mapping is not covered, so we map them ourselves ([extraction](https://docs.falkordb.com/graphrag/extraction.md)).
- **Retrieval:**
  - The default is `MultiPathRetrieval`.
  - Text-to-Cypher is opt-in (`enable_cypher=True`) and adds ~1.5 s.
  - Text-to-Cypher alone scored 53 %; adding it raised the total from 83.8 % to 84.2 % ([retrieval](https://docs.falkordb.com/graphrag/retrieval.md)).

**Text-to-Cypher tooling:**
- **FalkorDB/text-to-cypher** ([repo](https://github.com/FalkorDB/text-to-cypher)):
  - A Rust library and REST service. `POST /text_to_cypher` streams over SSE. There is also `POST /graph_query`, an MCP server on 3001 and REST on 8080.
  - Uses the `genai` crate, which lists Ollama and Gemini among its providers.
  - Configured with `DEFAULT_MODEL`, `DEFAULT_KEY` and `FALKORDB_CONNECTION`.
  - The model-string format is inconsistent in the README [unverified for Ollama/Gemma].
- **FalkorDB Browser AI chat panel:** natural language → Cypher, with a local LLM via Ollama (`http://localhost:11434`) or LM Studio ([chat-panel](https://docs.falkordb.com/browser/ui/chat-panel.md)). Handy for a quick demo.
- **QueryWeaver is Text2SQL** (PostgreSQL/MySQL) and does not fit our use ([QueryWeaver](https://github.com/FalkorDB/QueryWeaver)).
- **Graphiti** (`pip install graphiti-core[falkordb]`, `add_episode(...)`) is temporal episode memory; its docs show OpenAI only ([graphiti](https://docs.falkordb.com/agentic-memory/graphiti.md)).

### 2.6 FalkorDB Cloud

| Tier | Price | RAM | Notes |
|---|---|---|---|
| **Free** | free | 100 MB | AWS or GCP, standalone. **No TLS, no VPC, no persistence, no backups.** **Stopped after 1 day unused; deleted after 7 days; deleted data is unrecoverable.** |
| Startup | from $73/month | 1 GB | TLS, backups every 12 h |
| Pro | from $350/month | 8 GB | HA, multi-zone, AOF + snapshot |
| Enterprise | custom | custom | Azure, VPC, cluster, backups every 2 h |

Sources: [free tier](https://docs.falkordb.com/cloud/tiers/free), [overview](https://docs.falkordb.com/cloud/), [plans](https://www.falkordb.com/plans/), [features](https://docs.falkordb.com/cloud/features).

- **Unknowns:** what counts as "use", whether a stopped Free instance keeps its data, whether an India region exists, IP allowlisting, and whether PUBLISH/SUBSCRIBE is allowed — all [unverified].
- **Connecting** ([connect](https://docs.falkordb.com/cloud/console/connect)):
  - The `host:port` is on the Connectivity tab, and the port is often not 6379.
  - User `falkordb`; the password is on Instance Details.
  - Set `ssl=True` only on tiers with TLS.
- **Verdict:** the Free tier is unsafe for a judged URL. If you use it anyway, run a keep-alive plus a re-seed script.

### 2.7 Windows, Docker and persistence

- **Images** ([docker](https://docs.falkordb.com/operations/docker.html), [README](https://github.com/FalkorDB/FalkorDB)):
  - `falkordb/falkordb`: server plus Browser, on 6379 and 3000.
  - `falkordb/falkordb-server`: server only.
  - `falkordb/falkordb-browser`: Browser only; needs `FALKORDB_URL`.
- **No native Windows build exists.** Docker Desktop on WSL2 works on this laptop for 4.22.0 and 6.0.1 [team-measured] (`reference\FalkorDB\docs\falkordb-issues.md`).
- **Baked-in default:** `ENV FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"`. Setting `FALKORDB_ARGS` replaces the whole string, because `run.sh` appends it after the module load.
  - Confirmed in `master/build/docker/Dockerfile` (C engine) and `main/build/runtime/Dockerfile` (Rust) ([Dockerfile master](https://raw.githubusercontent.com/FalkorDB/FalkorDB/master/build/docker/Dockerfile), [Dockerfile main](https://raw.githubusercontent.com/FalkorDB/FalkorDB/main/build/runtime/Dockerfile), [run.sh](https://raw.githubusercontent.com/FalkorDB/FalkorDB/master/build/docker/run.sh)).
  - Under load, the default queue of 25 rejected 18 % of requests with "Max pending queries exceeded" [team-measured] (`reference\FalkorDB\docs\bench\README.md`).
- **Environment variables read by `run.sh`:**
  - `BROWSER` (default 1)
  - `TLS` (1 means self-signed certificates on 6379)
  - `REDIS_ARGS` (prepended)
  - `FALKORDB_ARGS` (appended)
  - Whether the 6.x image ships an identical `run.sh` is [unverified].
- **Data path:** `FALKORDB_HOME=/var/lib/falkordb`, with `/data` as a symlink to the data dir. Mount `/var/lib/falkordb/data` ([persistence](https://docs.falkordb.com/operations/persistence.html)).
  - Prefer named volumes, because 6.0.2 runs as non-root. Bind-mount permissions on Windows are [unverified].
  - ACL users are not persisted; use `--aclfile` on the volume.
- **Durability** ([durability](https://docs.falkordb.com/operations/durability.html)):
  - Balanced config: `save 900 1`, `save 300 10`, `save 60 1000`, `appendonly yes`, `appendfsync everysec`.
  - AOF wins on restart.
  - Back up graphs with `DUMP`/`RESTORE` or `GRAPH.COPY`.
  - The image's default persistence mode is [unverified].
- **Configuration** ([configuration](https://docs.falkordb.com/getting-started/configuration.html)):
  - Load-time only: `THREAD_COUNT`, `CACHE_SIZE`, `NODE_CREATION_BUFFER`, `BOLT_PORT`.
  - Runtime via `GRAPH.CONFIG SET`, **not persisted**: `TIMEOUT_MAX`, `TIMEOUT_DEFAULT`, `MAX_QUEUED_QUERIES`, `RESULTSET_SIZE`, `QUERY_MEM_CAPACITY`, `DELAY_INDEXING`, `JS_HEAP_SIZE`.

**Recommended local run.** The values are the team's earlier choice (`reference\FalkorDB\docker-compose.yml`). Do not run this now; Docker is stopped by rule.

```powershell
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 `
  -v falkordb_data:/var/lib/falkordb/data `
  -e REDIS_ARGS="--appendonly yes --appendfsync everysec" `
  -e FALKORDB_ARGS="TIMEOUT_DEFAULT 3000 TIMEOUT_MAX 10000 MAX_QUEUED_QUERIES 500 RESULTSET_SIZE 10000" `
  falkordb/falkordb:v4.22.0      # or :6.0.2 with the query rules in 2.1; never :latest
```

### 2.8 Live-write patterns

- **Concurrency** ([concurrency](https://docs.falkordb.com/design/concurrency.md)):
  - One writer per graph at a time (FIFO), with parallel snapshot-isolated readers.
  - Each query is atomic.
  - The docs advise batching changes into single queries.
  - With 100 writers, one graph gave a 260–360 ms write p50; 4 graphs gave 12–16 ms [team-measured] (`reference\FalkorDB\docs\bench\README.md`).
- **Ingest design:**
  - Micro-batch utterances (flush every N utterances or ~1–2 s) with `UNWIND $rows`. The cadence is a design suggestion [unverified throughput].
  - Use one graph per org or tenant if there are many writers.
- **MERGE:**
  - MERGE nodes separately, then the relationship. Merging a whole path can duplicate nodes ([merge](https://docs.falkordb.com/cypher/merge.md)).
  - Range-index the MERGE keys (`id`).
- **UNWIND:**
  - The docs show only literal lists ([unwind](https://docs.falkordb.com/cypher/unwind.md)).
  - Parameterized `UNWIND $rows` with `vecf32(r.embedding)` worked in the team's earlier project (`reference\FalkorDB\cypher\10_ingest_episode.cypher`).
  - On 6.0.x, follow the #3343 workarounds.

Suggested shape (untested on 6.0.2 [unverified]; it avoids WHERE-after-UNWIND):

```cypher
UNWIND $rows AS r
MERGE (m:Meeting {id: r.meeting_id})
MERGE (s:Speaker {id: r.speaker_id})
MERGE (u:Utterance {id: r.id})
  ON CREATE SET u.text = r.text, u.t_start = r.t0, u.t_end = r.t1, u.embedding = vecf32(r.emb)
MERGE (u)-[:SPOKEN_BY]->(s)
MERGE (m)-[:HAS_UTTERANCE]->(u)
```

- **Bulk load:** `GRAPH.BULK` / falkordb-bulk-loader can only create new graphs. That suits seeding a demo graph, not live ingest ([bulk-spec](https://docs.falkordb.com/design/bulk-spec.md)).
- **Change notifications** [corrected]:
  - FalkorDB documents **no trigger, CDC-out or graph-change feed**. The only CDC material is for loading SQL sources into FalkorDB ([docs index](https://docs.falkordb.com/_llms/falkor-db/documentation.txt), [sql-to-falkordb](https://docs.falkordb.com/operations/migration/sql-to-falkordb)).
  - Live UI updates must come from our app:
    - (a) **poll** by `t_start > last_seen`; or
    - (b) the ingest writer `PUBLISH`es `{meeting_id, utterance_ids, speaker, t}` (self-hosted FalkorDB runs inside redis-server 8.10.2), and a subscriber fans out over SSE or WebSocket. The team did this before (`reference\FalkorDB\packages\hiveboard\hiveboard\events.py`).
  - A long-running host (Render, Railway, Fly) is the **simpler** home for the subscriber, **but not strictly required**:
    - Vercel Functions support SSE and native WebSockets (public beta, all plans, Fluid compute). Connections close at max duration (300 s on Hobby), so clients need reconnect, resubscribe and reload logic.
    - Vercel recommends Redis pub/sub or Redis Streams for cross-instance fan-out ([Vercel WS KB](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections), [Vercel pub/sub](https://vercel.com/kb/guide/publish-and-subscribe-to-realtime-data-on-vercel)).
  - Pub/sub on FalkorDB Cloud is [unverified].

### 2.9 Licence [corrected]

- **The FalkorDB server is SSPLv1** ([license page](https://docs.falkordb.com/references/license.md)).
  - On the default `main` branch (Rust 6.x), the LICENSE file holds the full SSPLv1 text **followed by the full Elastic License 2.0 text**, with no explanation, and GitHub reports `NOASSERTION` ([LICENSE main](https://raw.githubusercontent.com/FalkorDB/FalkorDB/main/LICENSE)).
  - The `master` (C engine) `LICENSE.txt` is SSPLv1 only.
- OSI states SSPL is not an open-source licence ([OSI, 19 Jan 2021](https://opensource.org/blog/the-sspl-is-not-an-open-source-license)). ELv2's OSI status was not checked [unverified].
- **Do not call FalkorDB "open source".** Our own repo can be MIT or Apache-2.0.
- SSPL section 13 obligations apply only if we offer FalkorDB's functionality to third parties as a service.

---

## 3. Voice stack

### 3.1 What the reference projects already prove (read only; do not copy code, see 5.4)

**who-said-what-desktop** (Windows, Python 3.12, NeMo) uses these models (`reference\who-said-what-desktop\wsw\engine\models.py`, lines 42–66):

| Role | HF repo id | Size | Licence (per code) |
|---|---|---|---|
| Diarizer (Sortformer, `SortformerEncLabelModel`) | `nvidia/Nemotron-3-Diarization` | 199 MB | OpenMDW-1.1 |
| Live English multi-speaker ASR | `nvidia/multitalker-parakeet-streaming-0.6b-v1` | 2.49 GB | NVIDIA Open Model |
| Live multilingual ASR (40 locales) | `nvidia/nemotron-3.5-asr-streaming-0.6b` | 2.37 GB | OpenMDW-1.1 |
| Offline ASR | `nvidia/parakeet-tdt-0.6b-v3` | 2.5 GB | CC-BY-4.0 |
| Voiceprints (naming speakers) | `nvidia/speakerverification_en_titanet_large` | 102 MB | CC-BY-4.0 |
| Optional LLM | `unsloth/Qwen3.5-4B-GGUF` | 2.74 GB | Apache-2.0 |

**Measured on this laptop** [team-measured]:
- **Live:** an AMI ES2002a 3-minute crop ran at **182 ms per 1.12 s chunk** (p95 233 ms) with **3.2 GB** peak VRAM (`reference\who-said-what-desktop\README.md` line 176). That peak is with **4 ASR streams**; more simultaneous speakers cost more VRAM (`docs\USAGE.md` line 141).
- **Offline:** a 21-min AMI meeting diarized in 6.5 s (RTFx ≈ 195) at 1.5 GB VRAM (`docs\RESEARCH.md`, "Measured here").
- **NeMo import** takes ~8 s warm and ~4 min on first import.

**Windows specifics:**
- NeMo must come from git main; the 3.0.0 PyPI wheel lacks the needed modules (`docs\RESEARCH.md` section 1).
- `flex_attention` + `torch.compile` needs triton, which has no Windows build. The workaround is an eager kernel (`wsw\engine\compat.py`).
- Torch cu126 wheels need no CUDA toolkit, only driver 525+ (`docs\INSTALL.md` line 18).
- Models download via `hf_hub_download` **without a token** (`wsw\engine\models.py` ~line 178).

**Other facts:**
- VAD is `silero-vad>=6.0`; the API is FastAPI with HTTP and WebSocket (`pyproject.toml`, `wsw\server\api.py` line 18).
- **who-said-what (Android)** uses sherpa-onnx 1.13.8, Silero VAD, Parakeet-TDT 0.6B v2 int8 ONNX, TitaNet-small and 3D-Speaker ERes2Net (`reference\who-said-what\THIRD_PARTY_NOTICES.md`). sherpa-onnx has no Sortformer support.

### 3.2 Streaming speaker diarization

| Option | Model id(s) | Licence | Gated? | Streaming | 8 GB GPU real time | CPU-only |
|---|---|---|---|---|---|---|
| **Nemotron 3 Diarization** (Sortformer) | `nvidia/Nemotron-3-Diarization` (~100M params, 99.2M per safetensors) | OpenMDW-1.1, commercial OK | Card says an HF token is needed; the desktop app downloads it without one | **Yes.** Up to 8 speakers; profiles 0.32 / 0.64 / **1.04 s (default)**, plus 30.4 s offline | **Yes**, proven here [team-measured] (the card itself gives no VRAM figure) | Card lists Linux and Ampere/Ada/Blackwell/Hopper GPUs. CPU [unverified]. Hosted: DigitalOcean `nemotron-3-diarization` (4.1). |
| Streaming Sortformer v2.1 | `nvidia/diar_streaming_sortformer_4spk-v2.1` | NVIDIA Open Model License | not stated | Yes, max **4** speakers. RTF 0.093 (1.04 s profile) on an RTX 6000 Ada | Likely (117M) [unverified on 4060] | [unverified] |
| pyannote community-1 | `pyannote/speaker-diarization-community-1` | CC-BY-4.0 | **Yes** (conditions + HF token) | **No streaming documented** [corrected: the cards do not say "offline-only"; their "offline use" means without internet] | Whole-file. 31 s per hour of audio on an H100 | Runs on CPU by default; speed [unverified] |
| pyannote 3.1 | `pyannote/speaker-diarization-3.1` | MIT | **Yes** | No streaming documented | Whole-file, pure PyTorch | Slow; speed [unverified] |
| **diart** | `pyannote/segmentation(-3.0)` + `pyannote/embedding` | MIT | Yes (via pyannote) | **Yes. Real-time and overlap-aware by design** (official implementation of "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation"). 500 ms–5 s latency; `diart.serve` WebSocket. [corrected] | [unverified] | Claims real time on CPU. **Stale:** last release 0.9.2 on 12 Feb 2025, Python 3.8–3.10. It suggests `pyannote.audio<3.1` only to reproduce its benchmark results. |
| WhisperX | Whisper + community-1 | BSD-2-Clause | Yes | **No** (batch only). Overlapping speech "not handled particularly well". | "<8GB" for large-v2; 70x real time in batch | `--device cpu`, slow |
| sherpa-onnx diarization | pyannote-seg-3.0 ONNX + 3D-Speaker/NeMo embeddings | Apache-2.0 runtime | No | Streaming [unverified] | n/a | Built for CPU/edge; Python + WASM APIs; RTF [unverified] |

Sources: https://huggingface.co/nvidia/Nemotron-3-Diarization , https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1 , https://huggingface.co/pyannote/speaker-diarization-community-1 , https://huggingface.co/pyannote/speaker-diarization-3.1 , https://github.com/pyannote/pyannote-audio , https://github.com/juanmc2005/diart , https://pypi.org/project/diart/ , https://github.com/m-bain/whisperX , https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html

**Verdict** [corrected]:
- **Nemotron-3-Diarization is the only option proven real time on this 8 GB laptop.**
- diart is overlap-aware and online by design, but it is stale, depends on gated pyannote models and is untested here, so it is not a reliable drop-in.
- On a CPU-only cloud host, do not attempt live diarization. A Render free instance (0.1 CPU / 512 MB) is too small for any torch pipeline ([compute plans](https://render.com/docs/compute-plans)).

### 3.3 Streaming ASR

| Option | Licence | Streaming | Footprint / quality | Notes |
|---|---|---|---|---|
| `nvidia/multitalker-parakeet-streaming-0.6b-v1` | NVIDIA Open Model License | Yes; chunks of 0.08 / 0.16 / 0.56 / 1.12 s (`att_context_size`) | 600M. One instance per speaker, driven by the diarizer. cpWER AMI-IHM 21.26 | Overlap-aware. The desktop default. https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | OpenMDW-1.1 | Yes; 80 ms–1.12 s | 600M, 40 locales. FLEURS avg WER 8.84 (19 locales, 1.12 s); en-US 7.91. NeMo 26.06; Linux listed | Indian accent coverage [unverified]. https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b |
| `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | Chunked via a NeMo script (2 s) | RTFx 3332.74; ≥2 GB RAM; 25 European languages | Best offline word timestamps. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| `nvidia/canary-1b-v2` | CC-BY-4.0 | Not a live option | 978M; ≥6 GB RAM | ASR + translation. https://huggingface.co/nvidia/canary-1b-v2 |
| faster-whisper | MIT | No built-in streaming (wrappers: WhisperLive, Whisper-Streaming, speaches) | large-v2: fp16 4525 MB, int8 2926 MB VRAM | Needs CUDA 12 cuBLAS + cuDNN 9 (Purfview bundle on Windows). https://github.com/SYSTRAN/faster-whisper |
| whisper.cpp | MIT | `whisper-stream` ("naive example") | large ≈ 3.9 GB | CUDA, Vulkan and **WASM**; built-in Silero VAD. https://github.com/ggml-org/whisper.cpp |
| **Gemma 4 native audio** | Apache 2.0 | **No streaming**; ≤30 s clips | E2B/E4B/12B only (1.2–1.3) | See 3.4 |

### 3.4 Where Gemma 4 audio fits, and the VRAM budget

- **Use Gemma 4 E2B/E4B (or 12B) audio for short spoken questions** (<30 s) in the voice-first Q&A loop, and optionally to interpret a diarized segment (≤30 s). This is the honest G2 story: "Gemma 4 native ASR/understanding on ≤30 s diarized segments and spoken questions" [inference].
- **Do not use it to transcribe whole meetings.** It does not stream, does not attribute speakers, and caps clips at 30 s.
- **VRAM budget** [calc]: 8 GB − 3.2 GB (live pipeline, 4 ASR streams) ≈ 4.8 GB.
  - E2B Q4 (2.9 GB weights) fits beside live diarization.
  - E4B Q4 (4.5 GB weights before KV cache) is too tight while the live pipeline runs. Load it between meetings, or answer on hosted 31B.
- **Runtimes that take audio:**
  - Transformers (E2B/E4B/12B).
  - llama.cpp `llama-server` with `input_audio` (E2B/E4B documented).
  - Ollama (flaky on Windows, 1.6).
  - LiteRT-LM CLI.
  - Check that any `llama-cpp-python` wheel bundles a llama.cpp new enough for the Gemma 4 audio projector [unverified] (`reference\who-said-what-desktop\docs\LLM.md`).

### 3.5 TTS (Gemma outputs text only)

| Option | Licence | Where it runs | Notes |
|---|---|---|---|
| **Browser `speechSynthesis`** (default) | Web standard, Baseline since Sept 2018 | Browser; free | Voices vary by device (`getVoices()`; handle `voiceschanged`). `cancel()` stops the queue, which enables barge-in. Network-backed voices [unverified per browser]. https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis |
| Kokoro-82M | Apache-2.0 weights | Python `kokoro>=0.9.2` (+ `espeak-ng`), 24 kHz | v1.0: 8 languages, 54 voices. Latency [unverified]. https://huggingface.co/hexgrad/Kokoro-82M |
| kokoro-js (optional upgrade) | Weights Apache; JS package licence [unverified] | `onnx-community/Kokoro-82M-v1.0-ONNX`; `device: "wasm"` or `"webgpu"`; `dtype` fp32/fp16/q8/q4/q4f16 (fp32 on WebGPU) | `TextSplitterStream` + `tts.stream()` speaks sentence by sentence. Download size [unverified]. https://github.com/hexgrad/kokoro/tree/main/kokoro.js |
| Piper | Active fork `OHF-Voice/piper1-gpl` is **GPL-3.0**. MIT `rhasspy/piper` was archived on 6 Oct 2025. | `pip install piper-tts` | Per-voice licences [unverified]. Weaker fit. https://github.com/OHF-Voice/piper1-gpl , https://github.com/rhasspy/piper |
| DigitalOcean serverless TTS | Hosted (`qwen3-tts-voicedesign`, `fal-ai/elevenlabs/tts/multilingual-v2` via `/v1/audio/speech`) | Hosted | Pricing [unverified]. Closed models weaken a Track 2 story. https://docs.digitalocean.com/products/inference/details/models/ |

### 3.6 Browser capture and VAD

- **Secure context is mandatory.** `getUserMedia` works only on HTTPS, `file://` or `localhost`; elsewhere `navigator.mediaDevices` is undefined ([getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)). AudioWorklet is also secure-context-only ([AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)). **Never demo over `http://192.168.x.x`.**
- **Pipeline:**
  - `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true}})` → `AudioWorkletNode` → 16 kHz mono Float32 → 20–100 ms frames → binary WebSocket, or a whole-utterance POST.
  - `echoCancellation` is Baseline since Jan 2020 ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/echoCancellation)).
  - MediaRecorder (Opus/WebM) needs server-side decoding, and its chunks may not decode independently [unverified]. Use it only for "record a whole question, then upload".
- **Payload size** [calc]: 16 kHz mono 16-bit PCM is 32 KB/s, so a 30 s question is ~0.96 MB, under Vercel's 4.5 MB body cap.
- **VAD:**
  - **`@ricky0123/vad-web`** (and `vad-react`) runs Silero VAD (legacy, v5, v6) via onnxruntime-web.
    - `MicVAD` gives `onSpeechStart`, and `onSpeechEnd(audio)` returns a 16 kHz Float32Array utterance.
    - Pin `onnxWASMBasePath`/`baseAssetPath`; the quick start uses `vad-web@0.0.31` and `onnxruntime-web@1.22.0`.
    - Licence [unverified] ([vad](https://github.com/ricky0123/vad)).
  - **Silero VAD** (server side): MIT; under 1 ms per 30+ ms chunk on one thread; 8/16 kHz; ONNX available. A stray "CC BY-NC 4.0" README badge conflicts with this; LICENSE not checked [unverified] ([silero-vad](https://github.com/snakers4/silero-vad)).

### 3.7 Barge-in

- **While an answer plays, keep vad-web running.** On `onSpeechStart`:
  - call `speechSynthesis.cancel()` (or stop the kokoro-js `AudioBufferSourceNode`);
  - abort the streamed Gemma request (`AbortController`);
  - start a new question.
- **Preventing self-triggering:**
  - Use `echoCancellation:true`.
  - Raise the VAD threshold during playback (option names [unverified]).
  - Use a headset or close-talk mic on stage.

### 3.8 Windows and model-access gotchas

1. **Gated models:** pyannote community-1 and 3.1 (and so WhisperX diarization and diart) need accepted conditions plus an HF token. **Pre-download every model before the venue**; the Wi-Fi is unknown.
2. **NeMo:** install from git main; apply the triton/`flex_attention` workaround; expect a ~4 min first import (3.1).
3. **faster-whisper:** the pip-based NVIDIA libs are Linux-only, so use the Purfview DLLs ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)). WhisperX on a Windows GPU asks for CUDA toolkit 12.8 ([whisperX](https://github.com/m-bain/whisperX)).
4. **llama-cpp-python CUDA:** the cu125 wheel works with the torch cu126 runtime DLLs; `import torch` must come before `import llama_cpp` (`reference\who-said-what-desktop\docs\LLM.md`).

---

## 4. Deployment: the required live URL

The submission needs a public GitHub repo, a **working live deployment URL judges can test**, and the selected track plus problem statement (`docs\handbook\03-track-01-best-use-of-gemma-4.md`, `04-track-02-best-open-source-ai-project.md`).

### 4.1 Options

**Vercel** ([limits](https://vercel.com/docs/functions/limitations), [Python runtime](https://vercel.com/docs/functions/runtimes/python)):
- **Python runtime:** ASGI/WSGI (FastAPI) on Python 3.12 (default), 3.13 or 3.14. `pyproject.toml` + `uv.lock` work, and streaming responses are supported.
- **Max duration:** Hobby 300 s (default and max). Pro defaults to 300 s, max 800 s, with 1800 s in beta.
- **Body cap:** 4.5 MB (413 `FUNCTION_PAYLOAD_TOO_LARGE`).
- **Compute:** Hobby 2 GB / 1 vCPU; Pro up to 4 GB / 2 vCPU.
- **Bundle** [corrected]: 250 MB standard, **500 MB uncompressed for Python**. "Large functions" (Beta) allow up to **5 GB** on Node.js, Bun and Python with Fluid compute + Active CPU. New projects are eligible by default (env var `VERCEL_SUPPORT_LARGE_FUNCTIONS`).
- **No GPU.** So: no NeMo or live diarization, and Gemma only through hosted APIs.
- **WebSockets** ([WS KB](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections), [changelog](https://vercel.com/changelog/websocket-support-is-now-in-public-beta)):
  - Public beta on all plans; Python ASGI since 23 Jul 2026.
  - Connections close at max duration, so clients need reconnect logic.
  - Connections are pinned to an instance; shared state needs Redis or similar.
  - Idle connections bill provisioned memory.
- **Big uploads:** go browser → Vercel Blob with a function-issued token ([KB](https://vercel.com/kb/guide/how-to-bypass-vercel-body-size-limit-serverless-functions)).
- **TCP:** use the Node.js runtime, which has TCP. The Edge runtime has no raw sockets and is deprecated ([SMTP/runtime KB](https://vercel.com/kb/guide/serverless-functions-and-smtp)).
- **Fit:** UI, REST/streamed Q&A and graph reads.

**Render:**
- **Free web service** ([free](https://render.com/docs/free)):
  - Spins down after 15 min without traffic; spin-up takes ~1 min.
  - 750 instance hours per month.
  - Free Postgres expires after 30 days.
- **Plans** ([compute plans](https://render.com/docs/compute-plans)):
  - Free is 0.1 CPU / 512 MB, then `0.5c-512mb` (starter, ~$7/month [unverified]), `1c-2g` … `12c-96g`.
  - **No GPU plans** ([community](https://community.render.com/t/does-render-offer-gpus/11222/1)).
- **WebSockets:** **no maximum duration**; connections drop on deploy ([websocket](https://render.com/docs/websocket)).
- **Disks** ([disks](https://render.com/docs/disks)):
  - Only on **paid** web services, private services or workers.
  - A service with a disk is single-instance, with no zero-downtime deploys.
  - For FalkorDB, mount at `/var/lib/falkordb/data`. No Render-specific FalkorDB guide exists [unverified].
- **Fit:** a long-lived API or WS relay plus FalkorDB with a disk, on paid plans during judging.

**Railway:**
- Verified FalkorDB templates with volumes, `FALKORDB_PRIVATE_URL` / `FALKORDB_PUBLIC_URL` and a **TCP Proxy** for external access.
- Login is user `default` with `FALKORDB_PASSWORD`. Pricing [unverified] ([railway](https://docs.falkordb.com/operations/railway.md)).

**DigitalOcean (Presenting Partner):**
- **Serverless Inference** (base `https://inference.do-ai.run`; `/v1/chat/completions`, `/v1/responses`, `/v1/audio/speech`) ([endpoints](https://docs.digitalocean.com/products/inference/how-to/si-endpoints/)):
  - `gemma-4-31B-it` at $0.18 / $0.50 per 1M ([pricing](https://docs.digitalocean.com/products/inference/details/pricing/)).
  - `nemotron-3-diarization` at **$0.00058 per second of audio**. It uses a dedicated diarization endpoint whose path is **not** in the documented endpoint list [unverified API shape].
  - **No ASR model is listed** ([models](https://docs.digitalocean.com/products/inference/details/models/)).
  - **Prepaid only**: it needs credits or a positive balance. Hackathon credits [unverified].
- **GPU Droplets** (on-demand, pricing effective 1 Aug 2026) ([GPU pricing](https://www.digitalocean.com/pricing/gpu-droplets)):
  - RTX 4000 Ada 20 GB **$0.76/h**; RTX 6000 Ada 48 GB $1.57/h; L40S 48 GB $1.57/h; H100 80 GB $4.41/h.
  - **BLR1 (Bengaluru)** is a GPU datacenter, but which GPU types it offers is [unverified] ([availability](https://docs.digitalocean.com/products/droplets/details/availability/)).
  - Whether a powered-off droplet still bills is [unverified].
- **App Platform:**
  - 3 static sites free; containers from $5/month (1 shared vCPU, 512 MiB) ([pricing](https://docs.digitalocean.com/products/app-platform/details/pricing/)).
  - WebSockets supported ([tutorial](https://docs.digitalocean.com/developer-center/deploy-an-app-using-websockets-to-app-platform/)).
  - No GPU ([limits](https://docs.digitalocean.com/products/app-platform/details/limits/)).

**Hugging Face Spaces:**
- **Creating Gradio/Docker Spaces now needs a paid plan.** Static Spaces are free. Free accounts older than 30 days with a verified email can host **up to 2 ZeroGPU Spaces** ([overview](https://huggingface.co/docs/hub/spaces-overview)).
- **ZeroGPU** ([zerogpu](https://huggingface.co/docs/hub/spaces-zerogpu)):
  - Half an RTX Pro 6000 Blackwell (48 GB). **Gradio only.** No `torch.compile` (AOT only).
  - `@spaces.GPU` defaults to 60 s.
  - Daily quota: unauthenticated 2 min, free 5 min, PRO 40 min; extra time is $1 per 10 min.
  - Python 3.12.12 / 3.10.13; PyTorch 2.8–2.13.
  - Whose quota is charged when our backend calls the Space via `gradio_client` is [unverified].
- **Paid hardware:** T4 small $0.40/h, L4 $0.80/h, A10G small $1.00/h, L40S $1.80/h ([GPUs](https://huggingface.co/docs/hub/spaces-gpus)).
- **Fit:** a free ZeroGPU Space running `google/gemma-4-E4B-it` with audio as a "spoken question → text" endpoint. NeMo streaming fits poorly.

**Google Cloud Run GPU** ([GPU](https://docs.cloud.google.com/run/docs/configuring/services/gpu), [Gemma guide](https://docs.cloud.google.com/run/docs/run-gemma-on-cloud-run)):
- **L4 24 GB** (needs ≥4 CPU / 16 GiB) in `asia-south1` (Mumbai, **invitation only**), `asia-southeast1`, `europe-west1/4`, `us-central1` and `us-east4`. RTX PRO 6000 96 GB is also available.
- Scales to zero; instances with drivers start in ~5 s, and model load is extra. New projects get 3 GPUs of quota per region.
- Google's Gemma 4 guide uses a ≥240 s startup probe, so judges hitting a cold instance would wait minutes.
- L4 price [unverified].

**FalkorDB Cloud:** see 2.6. The Free tier is unsafe for judging; Startup ($73/month, TLS) is a paid option.

### 4.2 Recommended topology [inference]

The design rule: **the judge-facing URL must work without the laptop.** The laptop adds live ears on stage.

```
[Presenter laptop, RTX 4060 8 GB]                         [Cloud: always on]
mic/room -> our speech worker (rewritten in window)      Vercel Hobby (HTTPS, free)
  Nemotron-3-Diarization (streaming, 1.04 s)               - Web UI (static)
  + multitalker Parakeet 0.6B (English)                    - FastAPI on Python Fluid functions:
  + TitaNet voiceprints (naming)                             POST /api/ingest   (UNWIND batches, <4.5 MB)
  -> speaker+timestamp utterances --HTTPS batches-->         POST /api/ask      (streamed answer + citations)
  optional: Gemma 4 E2B Q4 audio (llama.cpp, ~2.9 GB)        GET  /api/meetings, /api/graph
     for spoken questions on stage                           GET  /api/live?since=t  (poll or SSE + reconnect)
                                                          FalkorDB, pinned tag, persistent volume, reachable over TCP
                                                            (Railway template + TCP proxy, or FalkorDB Cloud Startup)
                                                          Gemma 4 31B text: Gemini API or DigitalOcean (see 4.3)
                                                          HF ZeroGPU Space: gemma-4-E4B-it audio (spoken-question ASR)
Judge's browser: vad-web (Silero WASM) -> 16 kHz utterance -> /api/ask-voice -> answer text
                 -> speechSynthesis (or kokoro-js); barge-in cancels TTS + aborts the stream
```

**Why this shape:**
- Vercel gives HTTPS, so mic capture works (3.6).
- **Polling or SSE-with-reconnect** for the live view sidesteps the 300 s Hobby cap and needs no FalkorDB pub/sub (2.8).
- Gemma 4 31B is reachable without our own GPU.
- The judge's browser does VAD and TTS for free.

**Ingest path:**
- The laptop posts batches to `/api/ingest`, which writes them to FalkorDB with `UNWIND`.
- Alternatively, the laptop writes to FalkorDB directly when the DB port is reachable.
- Do not run per-utterance queries; writes are serialized per graph (2.8).

**FalkorDB host:**
- **Vercel must reach the DB over TCP**, so the DB needs a public TCP endpoint with a password. Railway's TCP proxy or FalkorDB Cloud both provide one.
- If FalkorDB goes on Render, put the API on Render too. Whether a Render-hosted Redis-protocol port is reachable from Vercel is [unverified].

### 4.3 Which Gemma 4 provider, by track [inference]

- **Track 1 (Best Use of Gemma 4):**
  - The MLH challenge text describes access **through the Gemini API** ([Track 1 challenge](https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-a0/challenges/01a03b87-be22-9180-5e2d-b3beeb66666d)), so make `gemma-4-31b-it` on the Gemini API primary.
  - Its free-tier data is used to improve Google products ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). Use only consented or demo meetings.
  - For G2, show **native Gemma 4 audio** (E2B/E4B via ZeroGPU or the laptop) on spoken questions.
  - No formal rule forbids local-only Gemma.
- **Track 2 (Best Open-Source AI Project, sponsor DigitalOcean):**
  - Make DigitalOcean `gemma-4-31B-it` primary. It is paid; DO's data-use terms were not checked [unverified].
  - Requirements: a public repo with an OSI licence, and open-weight AI central to the project. Gemma 4 is Apache 2.0; the NVIDIA speech models are OpenMDW / NVIDIA Open Model License / CC-BY-4.0.
  - Whether judges count the "NVIDIA Open Model License" as open-weight is [unverified].
  - Do not call FalkorDB open source (2.9).
- **Either track:** wrap the LLM behind one adapter with an env-var switch: Gemini API ↔ DigitalOcean ↔ Cloudflare `@cf/google/gemma-4-26b-a4b-it`.

### 4.4 Cost [calc]

- **Vercel Hobby:** $0.
- **DigitalOcean 31B:** an 8k-input + 400-output answer costs ≈ $0.0016, so 1,000 answers cost ≈ $1.60. The minimum prepaid top-up is [unverified].
- **Gemini API:** $0.
- **HF ZeroGPU:** $0, if the account is older than 30 days.
- **FalkorDB Cloud Startup:** $73/month. Railway pricing is [unverified].
- **Optional all-cloud live audio:** a DigitalOcean RTX 4000 Ada at $0.76/h ≈ $18/day. Run it only on 18 Oct, and keep pre-ingested mode as the always-on URL.

### 4.5 Fallbacks

1. **Laptop absent or broken:**
   - Before submission, run 2–3 meetings through the laptop pipeline (offline profile, RTFx ≈ 195) and ingest them into the hosted FalkorDB.
   - The URL still serves who-said-what browsing, graph views and Q&A with speaker + timestamp citations.
   - Spoken questions go through ZeroGPU, with a typed box as the last resort.
2. **Vercel ↔ DB connectivity fails:** move the FastAPI app to one paid Render web service co-located with FalkorDB on a disk. Render enforces no WebSocket duration limit.
3. **Gemma provider quota or outage:** flip the adapter env var (Gemini ↔ DigitalOcean ↔ Cloudflare 26B).
4. **DB loss** (e.g. a Free-tier deletion): keep a re-seed script; `GRAPH.BULK` can create a fresh graph.
5. **Hosted file upload (stretch goal):**
   - Flow: browser → Vercel Blob → DigitalOcean `nemotron-3-diarization` (API shape [unverified]).
   - It still needs hosted ASR, and DigitalOcean lists none.

### 4.6 Deployment gotchas

- **Mic needs HTTPS or localhost.** Use the Vercel URL or `localhost` on stage.
- **Vercel:**
  - 4.5 MB body cap: chunk ingest batches, and send big audio to Blob.
  - Hobby has a 300 s cap per invocation and per WebSocket.
  - WebSockets are beta.
- **Render free:** the first judge after 15 idle minutes waits ~1 min. Use a paid plan, or keep the judge-facing parts on Vercel.
- **DigitalOcean serverless is prepaid.** Top up before 18 Oct.
- **Cloud Run GPU:** the cold start needs a ≥240 s probe; `asia-south1` L4 is invitation-only.
- **Gemini API rate limits are shown only in AI Studio** ([rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)). A third-party figure of 15 RPM / 1,500 RPD is [unverified].

---

## 5. Event extras (no verifier pass: single-source)

### 5.1 MLH event page

Our event is **event 15112, slug `hacktoberfest-hack-day-bengaluru-a0`** ([event page](https://events.mlh.com/events/15112-hacktoberfest-hack-day-bengaluru-krowdkraft), [mlh.com](https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-a0), [API](https://hacktoberfest-api.mlh.com/api/events)).
- **Basics:**
  - Sun 18 Oct 2026, 8:30 AM–6:30 PM IST, at Startup Park, Koramangala, Bengaluru 560068. The API gives `2026-10-18T03:00:00Z` to `13:00:00Z`.
  - Free, in person, teams of 1–3.
  - Check in on OrganizerHQ. Bring a laptop, charger and government or institutional ID.
  - Registration closed 7 Oct, and the shortlist was due 9 Oct.
- **Separate processes:** registration is subject to a team idea-submission and RSVP process. One person per team must complete the Team Formation Form.
- **Tentative schedule:** build starts 9:30; lunch 12:30–1:30; **submission deadline 3:30 PM**; judging 3:45–5:30; final evaluation 5:30–6:00; winners at 6:00.
- **Two WhatsApp links:** the MLH page and the handbook (`docs\handbook\00-handbook.md` line 51) give different group links. Ask the organizers which is current.
- **Name clash:** another Hack Day, **"Hacktoberfest Hack Day Bengaluru x OLake"** (event 15468), runs the same day. Always use event 15112.

### 5.2 Challenges (prize categories)

- **Best Use of Gemma 4** ([challenge](https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-a0/challenges/01a03b87-be22-9180-5e2d-b3beeb66666d)):
  - Sponsor shown as "Deepmind Google".
  - Describes Gemma 4 as open-weight multimodal models reached **through the Gemini API**.
  - Suggested directions (not requirements): a multimodal assistant, a focused tool, or rapid prototyping with the Gemini API.
  - **No formal requirements, criteria or prize amount** are listed.
- **Best Open-Source AI Project** (sponsor DigitalOcean) ([challenge](https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-a0/challenges/01a0d987-0c41-12ea-13ac-5b85755dc2f3)):
  - Prize: an exclusive Hacktoberfest 2026 Winner DEV badge plus an MLH swag bag.
  - **Requirements:**
    - (1) open-source or open-weight AI must be an important part of the project;
    - (2) a **public GitHub repo with an open-source licence**;
    - (3) agent-skill entries must follow the Agent Skill Open Standard (no link given; likely https://agentskills.io/ [unverified]);
    - (4) model-harness entries need an original or meaningfully changed implementation.
- Both challenge pages currently say submissions are not open.
- **Local prizes are handbook-only, not on MLH pages** (`docs\handbook\00-handbook.md` lines 224–252):
  - Arduino Uno Q per member for the overall, Track 1 and Track 2 winners.
  - An overall cash prize, still TBA.
  - ₹500 vouchers for the ShipWright top 3.

### 5.3 Submission platform: not Devpost

- No page names Devpost.
- **Process:** one teammate opens **Challenges → Add Submission** on the Fest's MLH event page and submits before the host's deadline ([Hacktoberfest FAQ](https://hacktoberfest.com/llms-full.txt)).
- **Inference [unverified]:** the handbook's "portal TBA" is the MLH OrganizerHQ challenges page, possibly with extra KrowdKraft fields for the live URL, track and problem statement.
- **Judging criteria are not published** for this event (`00-handbook.md` line 200). The FAQ says the host picks winners from challenge rules and demos.

### 5.4 MLH rules on pre-existing code (decision-relevant)

**MLH Standard Hackathon Rules** ([rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)). Organizers may adopt, edit or replace these; KrowdKraft's rules take precedence.
- **Rule 6:** all work should be done during the hackathon period.
- **Rule 7:** a pre-existing idea is fine.
- **Rule 8:** ideas that have already been done are fine.
- **Rule 9: you may work on an idea from before, but not re-use its code or project materials.**
- **Rule 10:** libraries, frameworks and open-source code are allowed. Open-sourcing pre-event work just to reuse it is not.
- **Rule 11:** stop at time-up; small demo bug fixes only.
- **Rules 12–13:** code must be public and stay public after the event.
- **Rule 14:** organizers may disqualify at their discretion.
- **AI tools** are allowed but must be disclosed and listed in the submission.
- **Default judging criteria** (MLH categories at in-person events): **Technology, Design, Completion, Learning**, weighted equally.
  - Live demos are preferred over pitches.
  - Code quality, pitch quality, idea quality and how well it solves a problem are explicitly **not** criteria.

**KrowdKraft's window:** "Projects must be built between 13 and 18 October 2026" (`docs\handbook\03-…md`, `04-…md`). Teams may build before arriving at the venue, but only within that window (`00-handbook.md` line 188).
- The Hacktoberfest FAQ says Hack Day projects are completed and submitted on the day ([FAQ](https://hacktoberfest.com/llms-full.txt)). The host's window governs here.

**Organizer guidance:**
- **Rules template** ([rules template](https://guide.mlh.com/general-information/judging-and-submissions/rules-for-your-hackathon)):
  - No pre-event work and no reused code.
  - List frameworks in the README.
  - Uncredited AI use leads to disqualification and a report to cheating@mlh.io.
  - The project must not be a reskin of an existing AI tool.
- **Judging plan** ([judging plan](https://guide.mlh.com/general-information/judging-and-submissions/judging-plan.md)):
  - Expo (science-fair) judging.
  - About 2 min demo + 1 min questions per team.
  - 3/2/1 stack ranking, no formal rubric, with a focus on learning.
- **Cheating check** ([cheating check](https://guide.mlh.com/general-information/judging-and-submissions/cheating-check.md)). These flags prompt a closer look:
  - one big commit at the start of the event;
  - work started before the event;
  - pushes after the deadline;
  - a project that looks like an old portfolio project;
  - the same project submitted elsewhere.
  - A single final commit alone is not grounds for disqualification.

**Implications for us [inference]:**
- **Do not copy code from `reference\who-said-what*` or `reference\FalkorDB`.** Re-implement within 13–18 Oct, using public models and libraries (NeMo, falkordb-py) as dependencies.
- Disclose the earlier projects as prior work and inspiration in the README.
- Keep a granular commit history starting on or after 13 Oct.
- List AI tools used (e.g. Claude Code) in the README and the submission.

### 5.5 ShipWright (Technology Partner)

- **GitHub App:** https://github.com/apps/shipwright-agent (developer `anurag6569201`).
  - Its website link https://github.com/kpuru88/shipwright-agent returns 404.
  - The leaderboard links to https://shipwright.dev, which fails DNS.
- **Leaderboard:** https://shipwright-agent.onrender.com/leaderboard, with data from https://shipwright-agent.onrender.com/leaderboard.json. It is **ranked by "Fixed after review"**: merged PRs that got a fix push after ShipWright's review.

**Snapshot** (`generated_at` 2026-10-14T22:47:34Z, `since` 2026-10-13T17:30:00):
- Goal 600. Totals: 15 PRs opened, 15 reviewed, **11 counted**, 9 acted on.
- Review comments: 423 posted, 190 fixed, 646 skipped (fix rate 0.449).

| # | Repo | PRs | Counted | Fixed after review |
|---|---|---|---|---|
| 1 | Larry8668/saint-elms-fire-app | 7 | 6 | 6 |
| 2 | Sriram-Nambiar/JanVaani | 5 | 4 | 3 |
| 3 | Beingpanditt000/VisionCraft | 1 | 1 | 0 |
| 4 | AkifAhmedShariff/insightforge | 1 | 0 | 0 |
| 5 | pixelGoutham/PixelVision | 1 | 0 | 0 |

Competition for the top-3 ₹500 vouchers is thin.

**Handbook rules** (`docs\handbook\01-shipwright-leaderboard-challenge.md` lines 48–67):
- PRs opened on or after 13 Oct, in a public repo with the app installed.
- PRs must be human-created, reviewed and merged, with branches in the team repo (not forks).
- Fixes after review rank higher; `/shipwright rerun` retries a review.
- Use the same repo URL in the final submission.
- Artificial PR activity can lead to disqualification (`00-handbook.md` line 259).

### 5.6 hackFront India 2026 (Where U Elevate, "Innovation Partner")

- **Page JSON** ([page](https://whereuelevate.com/drills/hackfrontindia-2026)):
  - ONLINE, team-based, **team size 2–5**, status LIVE.
  - Runs 2026-07-06 to 2026-12-27.
  - **Registration ends 2026-10-23T18:29Z** (23 Oct 23:59 IST).
  - Prize value "Rs. 1 Crore +"; 4,672 registrations and 1,745 teams.
- **Press coverage conflicts** with the page: it reports a ₹5 lakh+ cash prize, a 36-hour Delhi finale and a 30 Sept registration close. Not resolved ([newswav](https://newswav.com/article/as-bharat-innovates-goes-global-hackfront-india-launches-to-power-the-next-A2606_W78AZn), [bisinfotech](https://www.bisinfotech.com/?p=193380)).
- **Link to our event:** the public page never mentions Hacktoberfest, MLH or KrowdKraft, and does not feed this Hack Day's judging.
  - Only the handbook lists registration as a pre-venue checklist item (`00-handbook.md` lines 309, 312).
  - A solo builder would need a hackFront teammate, because its minimum team size is 2.
  - Whether it is mandatory is [unverified]; ask the organizers.

### 5.7 Hacktoberfest 2026

Sources: https://hacktoberfest.com/ , https://hacktoberfest.com/questions/ , https://hacktoberfest.com/llms-full.txt

- **Format:**
  - Runs throughout October, run by MLH and DEV with DigitalOcean as presenting partner.
  - Focus on open-source AI and open-weight models.
  - Hundreds of one-day Fests: Hack Days and Meetups.
- **PRs no longer count toward Hacktoberfest rewards**, because of spam. ShipWright PRs matter only for the local leaderboard.
- **Hack Day rules:**
  - The host sets most of the details; projects are completed and submitted on the day.
  - Submission is through Challenges → Add Submission.
  - Prizes cover up to 4 people, but KrowdKraft's 1–3 team limit is stricter and applies here.
- **Rewards:** an MLH+DEV swag bag for Hack Day winners; a participation certificate; DEV badges (needs DEV linked to My Hacktoberfest); a sticker pack.
- **Sponsors include** Gemma, Google Cloud, Render and ElevenLabs.
- **Fests API:** https://hacktoberfest-api.mlh.com/api/events (539 events at fetch time).

### 5.8 Gemma links from the event page

- **Short-link targets:**
  - `mlh.link/gemma` → https://www.mlh.com/partners/gemma
  - `mlh.link/gemma-quickstart` → https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api
  - `mlh.link/gemma-docs` → https://ai.google.dev/gemma/docs/get_started
  - `mlh.link/gemma-beginnerguide` → https://github.com/google-gemma/cookbook
- **The quickstart:**
  - Lists `gemma-4-31b-it` and `gemma-4-26b-a4b-it`; its code samples use `gemma-4-26b-a4b-it`.
  - The key comes from AI Studio.
  - Audio input appears only in a banner, with no API example, so it is [unverified].
- **The MLH partner page:** Gemma 4 is Apache 2.0, accessed via the Gemini API, Vertex AI or locally.

### 5.9 Pages that failed

- https://github.com/kpuru88/shipwright-agent: 404.
- https://shipwright.dev: DNS failure.
- `guide.mlh.io/general-information/judging-and-submission` (singular): HTTP 500. The plural path works.
- The ShipWright leaderboard HTML shows "Loading…"; use the JSON.
- The hackFront page is client-rendered; only its metadata JSON is readable.

---

## 6. Fact-check ledger

The verifier ran on 30 decision-critical claims: **20 confirmed, 10 partially_true, 0 refuted.** The section 5 claims had no verifier pass.

| # | Claim | Verdict | Correction | Evidence URL |
|---|---|---|---|---|
| G1 | Native audio input only on Gemma 4 E2B, E4B and 12B Unified (26B A4B and 31B text+image). Tasks are ASR and AST; max 30 s per clip. | confirmed | — | https://ai.google.dev/gemma/docs/core/model_card_4 |
| G2 | Audio must be 16 kHz mono float32 in [-1,1]; 25 tokens/s; the guide covers multi-clip understanding with `pipeline('any-to-any')` on E2B/E4B/12B-it. | confirmed | — | https://ai.google.dev/gemma/docs/capabilities/audio |
| G3 | Gemma 4 is Apache 2.0 (not the Gemma Terms of Use); all variants output text only. | confirmed | — | https://ai.google.dev/gemma/docs/core/model_card_4 |
| G4 | The Gemini API serves only `gemma-4-31b-it` and `gemma-4-26b-a4b-it` (no audio). Free tier free, paid tier "Not available", free-tier data used to improve products. | confirmed | — | https://ai.google.dev/gemini-api/docs/pricing |
| G5 | No serverless API serves an audio-capable Gemma 4. HF Providers: none for E2B/E4B/12B. Vertex MaaS 26B only, no audio. Cloudflare, Together, DigitalOcean and OpenRouter serve non-audio variants only; Groq none. | partially_true | **Together removed `google/gemma-4-31B-it` from serverless on 2026-09-15** (replacement `zai-org/GLM-5.3-Flash`), so Together has no Gemma 4 serverless. The rest holds: HF mapping empty for E2B/E4B/12B-it; 26B-A4B-it on novita, featherless-ai, scaleway, deepinfra; Vertex MaaS 26B only (Experimental, 3 Apr 2026, text+image, no audio/video, **no thinking**); Cloudflare 26B no audio; DigitalOcean 31B serverless; OpenRouter 26B inputs are image/text/video; Groq none. The universal negative is unprovable: Eden AI lists deepinfra E4B "Audio input" but the deepinfra page 404s [unverified]; OpenRouter 31B modalities not checked [unverified]. | https://docs.together.ai/docs/deprecations |
| G6 | Ollama tags (e2b, e4b = latest, 12b, 26b, 31b, plus qat/q4_K_M/q8_0/bf16; e4b-it-qat 6.1GB, 12b-it-qat 7.2GB); Input column "Text, Image"; the only official audio note is v0.33.3 (MLX), so Windows/CUDA audio is unverified. | partially_true | Tags, sizes, e4b = latest and the "Text, Image" Input column on /tags are correct; more tags exist (cloud, -mlx, -mlx-bf16, -mxfp8, -nvfp4, 26b-a4b, mtp). **Wrong that v0.33.3 is the only signal:** the library page lists "Text, Image, Audio" for E2B/E4B, a ~300M audio encoder and an `audio` tag. v0.20.0-rc1 reads "support new Gemma4 audio_tower tensor naming (#15221)". User reports: #16584 (Win11, RTX 3090) e4b audio worked on v0.20.7, regressed on v0.30.6 (thinking on by default), 12b works with thinking disabled; #15333 (v0.20.2, Windows) intermittent GGML crashes. **Flaky and version-dependent; test the exact version.** | https://ollama.com/library/gemma4 |
| G7 | llama.cpp supports Gemma 4 audio + vision for E2B and E4B only; llama-server accepts OpenAI `input_audio` (multimodal experimental). | partially_true | multimodal.md lists only the E2B/E4B GGUFs for audio (26B/31B vision only, 12B unlisted). The "experimental" label is in `tools/server/README.md`, which documents `input_audio` (data or url; mp3/wav/flac via miniaudio). "E2B/E4B only" is stale in code: `ggml-org/gemma-4-12B-it-GGUF` mmproj sets `clip.audio.projector_type` = `gemma4ua`; PR #24088 (merged 3 Jun 2026) fixed mtmd for 12B; issue #24084 was closed by #24091. 12B audio exists upstream but is undocumented [end-to-end unverified]. | https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/main/convert.log |
| G8 | Official weight memory (no KV cache): E2B Q4_0 2.9, E4B Q4_0 4.5 (SFP8 8.9, BF16 17.9), 12B 6.7, 26B 14.4, 31B 17.5 GB. On 8 GB, E2B/E4B Q4 fit, 12B Q4 is borderline, 26B/31B need offload. | confirmed | — | https://ai.google.dev/gemma/docs/core |
| G9 | Native function calling (tool tokens, `tools` via apply_chat_template) and configurable thinking (`<\|think\|>` / `enable_thinking=True`; `<\|channel>thought…<channel\|>`) on all five variants. | confirmed | — | https://ai.google.dev/gemma/docs/capabilities/thinking |
| G10 | 12B (3 Jun 2026) is encoder-free with 40 ms frames; targets 16 GB VRAM; lists diarization with no benchmark or API; the only 12B GPU speeds are LiteRT-LM 69 tok/s (4090) and 50 tok/s (5080 Windows); no 4060 numbers. | partially_true | The blog confirms the architecture, 40 ms frames (640 floats), the 16 GB target and diarization without benchmarks, but **has no speed figures**. 69/50 tok/s come from the litert-community 12B card (1024 prefill/256 decode, 4096 ctx; 4090 Linux 3548 prefill, 0.3 s TTFT; 5080 Windows 391 prefill, 2.5 s TTFT). **Not the only figures:** M4 Pro GPU 29, MacBook Air M4 GPU 15, Web (M4 Max) 26 tok/s; ~7.3–7.9 GB GPU memory for the 6883 MB model. "No API" is only partly true: the local OpenAI-compatible `litert-lm serve` and Model Garden/Cloud Run/GKE self-deploy exist; there is no hosted serverless API. No RTX 4060 figures (confirmed). | https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm |
| F1 | FalkorDB 6.0.2 (latest, Rust, 8 Oct 2026) has an open silent bug: UNWIND before MATCH…WHERE…MATCH returns 0 rows. Workarounds: a WITH barrier or an inline map. v4.22.0 (C) is the alternative. | partially_true | #3343 opened 14 Oct 2026, open, **no labels**. It reproduces on `6.0.2` and `:edge`, and EXPLAIN shows no Unwind op. It lists **three** workarounds: `WITH n` barrier, inline map `{id: x}`, or `WHERE id(n) IN $ids` without UNWIND. It **never mentions the C engine**: v4.22.0 is a real tag (2026-09-30, amd64/arm64) and v4.x.y/edge-c are the unchanged C engine, but whether C is free of this bug is [unverified]. 4.x and 6.0 cannot replicate together; 6.0 lacks A*, CCH and Leiden/Louvain. | https://github.com/FalkorDB/FalkorDB/issues/3343 |
| F2 | Rust engine: MATCH…WHERE then a traversing MATCH drops the WHERE; DETACH DELETE then deletes unmatched data (open, labelled rust; C correct). | confirmed | — | https://github.com/FalkorDB/FalkorDB/issues/2972 |
| F3 | Writes serialized per graph, parallel snapshot-isolated reads, atomic queries; batch changes into single queries. | confirmed | — | https://docs.falkordb.com/design/concurrency.md |
| F4 | The image bakes `FALKORDB_ARGS="MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000"`; overriding replaces it; data in `/var/lib/falkordb/data`, with `/data` a symlink. | confirmed | — | https://raw.githubusercontent.com/FalkorDB/FalkorDB/main/build/runtime/Dockerfile |
| F5 | Cloud Free: 100 MB, AWS/GCP, no TLS, persistence or backups; stopped after 1 idle day, deleted after 7. Startup from $73/month for 1 GB with TLS and 12-hourly backups. | confirmed | — | https://docs.falkordb.com/cloud/tiers/free |
| F6 | Vector index syntax (dimension, similarityFunction, M 16, efConstruction 200, efRuntime 10), dims 1–4096, `queryNodes(label, attr, k, vecf32($q)) YIELD node, score`; docs ambiguous on score; team measured distance on 6.0.1. | partially_true | Syntax confirmed; the optional params are Cypher-map entries with colons (e.g. `M:32`), not `=`. The `queryNodes` signature is confirmed. **On score, the docs are internally inconsistent rather than silent:** "a similarity score" with `ORDER BY score DESC` in every example, yet euclidean "0 = identical". If the score is a distance, DESC is wrong. The team's distance measurement was not reproduced by the verifier; re-check on the pinned version. Dims: 1–4096, but the FAQ says "any dimension". | https://docs.falkordb.com/cypher/indexing/vector-index.md |
| F7 | GraphRAG-SDK 1.4.0 is async and LiteLLM-first; no official Ollama/Gemini examples; custom `LLMInterface` for local models; Gemma via LiteLLM plausible but unverified. | confirmed | — | https://docs.falkordb.com/graphrag/providers.md |
| F8 | No trigger, change-feed or pub/sub. Live UI must use app-side PUBLISH + SSE/WS from a long-running backend, because Vercel functions cap at 300 s on Hobby. | partially_true | Confirmed: no trigger/CDC/change-feed (CDC only loads SQL into FalkorDB); Hobby max 300 s (Pro/Enterprise default 300, max 800). **Overstated:** a separate long-running backend is not strictly required. Vercel Functions support SSE and native WebSockets (public beta, all plans, Fluid compute); connections close at max duration, so clients need reconnect, resubscribe and reload; Vercel recommends Redis pub/sub or Streams for fan-out. A long-running host is simpler, not forced. Pub/sub on FalkorDB Cloud [unverified]. | https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections |
| F9 | npm `falkordb` 6.8.0 needs Node >=22 (machine has 20); Python `FalkorDB` 1.8.0 (2026-10-14) needs Python >=3.10 and redis>=8,<9, and supports `falkor://`/`falkors://` in sync and async `from_url`. | confirmed | — | https://registry.npmjs.org/falkordb/latest |
| F10 | The FalkorDB server is SSPLv1; OSI says SSPL is not open source; our code can be MIT/Apache. | partially_true | The docs and README say SSPLv1; the OSI statement is dated 19 Jan 2021. **Nuance:** the `main` (Rust 6.x) LICENSE holds full SSPLv1 **plus full Elastic License 2.0** text with no explanation, and GitHub reports NOASSERTION. The `master` (C) LICENSE.txt is SSPLv1 only. ELv2's OSI status not checked [unverified]. Do not call FalkorDB open source. SSPL §13 applies only if we offer FalkorDB as a service to third parties. | https://raw.githubusercontent.com/FalkorDB/FalkorDB/main/LICENSE |
| V1 | Gemma 4 audio only on E2B/E4B/12B; 30 s max; 16 kHz mono float32; 25 tok/s; no streaming documented. | confirmed | Precision: the model is named "12B Unified" (`google/gemma-4-12B-it`, ~11.95B). The audio page never mentions streaming; its mic example records a whole clip first. The model card marks 26B A4B and 31B as Audio: No. | https://ai.google.dev/gemma/docs/core/model_card_4 |
| V2 | The Gemini API serves only `gemma-4-31b-it` and `gemma-4-26b-a4b-it` (text/image); free tier only; free-tier data used to improve products. | confirmed | — | https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api |
| V3 | DigitalOcean Serverless Inference hosts `gemma-4-31B-it` (Chat Completions/Responses) at $0.18/$0.50 per 1M; prepaid only. | confirmed | — | https://docs.digitalocean.com/products/inference/details/pricing/ |
| V4 | DigitalOcean serves `nemotron-3-diarization` at $0.00058/s via a dedicated endpoint; no ASR model; endpoint path not documented. | confirmed | — | https://docs.digitalocean.com/products/inference/how-to/si-endpoints/ |
| V5 | Nemotron-3-Diarization (~100M, OpenMDW-1.1): streaming, up to 8 speakers, profiles 0.32/0.64/1.04 s; runs on this RTX 4060 with multitalker Parakeet at 3.2 GB peak and 182 ms per 1.12 s chunk. | confirmed | Precision: 99.2M params; also a 30.4 s offline profile; Linux preferred; Ampere/Ada/Blackwell/Hopper. **The card gives no VRAM figure and does not mention Parakeet.** Local numbers come from `reference\who-said-what-desktop\README.md` line 176 (AMI ES2002a 3-min crop) and `docs\USAGE.md` line 141 (3.2 GB with 4 ASR streams; more speakers cost more VRAM). | https://huggingface.co/nvidia/Nemotron-3-Diarization |
| V6 | Vercel Fluid: Hobby 300 s max; 4.5 MB body; 500 MB Python bundle; WebSockets (Python ASGI) in public beta, closing at max duration. | partially_true | Accurate: 300 s Hobby default and max; 4.5 MB (413 `FUNCTION_PAYLOAD_TOO_LARGE`); WS public beta on all plans; Python ASGI/WSGI since 23 Jul 2026; reconnect logic needed. **Correction:** 500 MB (uncompressed) is only the standard Python limit. "Large functions" (Beta) allow up to **5 GB** on Node.js, Bun and Python with Fluid compute + Active CPU; new projects are eligible by default (`VERCEL_SUPPORT_LARGE_FUNCTIONS`). Hobby is also capped at 2 GB memory and 1 vCPU. | https://vercel.com/docs/functions/limitations |
| V7 | Render has no GPUs; the free service (0.1 CPU/512 MB) spins down after 15 min and takes ~1 min to wake; no max WebSocket duration. | confirmed | — | https://render.com/docs/compute-plans |
| V8 | HF Gradio/Docker Spaces need a paid plan to create; free accounts >30 days old with verified email get up to 2 ZeroGPU Spaces (Gradio only, half an RTX Pro 6000 48 GB, 60 s default, daily quotas). | confirmed | — | https://huggingface.co/docs/hub/spaces-zerogpu |
| V9 | pyannote community-1 and 3.1 are gated and offline-only; diart's last release was Feb 2025 and it recommends pyannote.audio<3.1; WhisperX is batch-only; none is a real-time overlap-aware option. | partially_true | Holds: both pyannote models are gated; diart 0.9.2 (12 Feb 2025); WhisperX is batch/file-only, uses community-1 and says overlapping speech is "not handled particularly well". **Corrections:** (1) the pyannote cards never say "offline-only"; "offline use" means without internet, so the accurate claim is "no documented streaming". (2) diart's `pyannote.audio<3.1` is only to reproduce its results. (3) **diart IS real-time and overlap-aware** (official implementation of the overlap-aware online diarization paper). It is stale, not unsuitable by design. | https://github.com/juanmc2005/diart |
| V10 | getUserMedia and AudioWorklet need a secure context (HTTPS or localhost); never use a LAN IP over http. | confirmed | — | https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia |

**Section 5 (event extras):** 10 decision-critical claims were submitted (Track 2 requirements, Track 1 "via Gemini API" wording, Gemini API model IDs, OrganizerHQ submission, MLH rules 6/9/10, MLH criteria, cheating check, ShipWright snapshot, PRs no longer counting, hackFront dates). **The verifier returned no verdicts**, so treat them as single-source and re-check the live pages before relying on them.

**Open questions to resolve before 18 Oct:**
- Whether Ollama audio works on our exact version on Windows/CUDA (G6).
- Whether 12B audio works end to end in llama.cpp (G7).
- Gemma 4 E2B/E4B/12B speed on the RTX 4060.
- Whether 4.22.0 is free of #3343.
- Vector score direction on the pinned version.
- Pub/sub and India region on FalkorDB Cloud.
- Railway pricing, and whether Render can expose a TCP port to Vercel.
- The DigitalOcean diarization endpoint shape; any DigitalOcean hackathon credits.
- Whose ZeroGPU quota is charged when a backend calls the Space.
- Licences for vad-web and kokoro-js.
- Which WhatsApp group is current.
- Whether hackFront registration is mandatory.
- The submission portal's fields.
- Whether judges accept the NVIDIA Open Model License as "open-weight".
