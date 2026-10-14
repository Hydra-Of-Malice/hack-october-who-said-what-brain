# Repo audit: reference repositories for "Hack October - Who Said What Brain"

- **Audit date:** 15 Oct 2026 (IST). Hacktoberfest Hack Day Bengaluru 2026: development window 13-18 Oct 2026, on-site day Sun 18 Oct, submissions close 3:30 PM IST.
- **Scope:** the three read-only clones under `D:\HACKATHONS\hackoctober fest\reference\`: `who-said-what`, `who-said-what-desktop` and `FalkorDB`.
- **Method:** this file merges four audit passes. There was one pass for each who-said-what repo and two for `FalkorDB` (product/UI and engine). Commit ranges, the assumption searches and every contradiction between passes were re-checked against the clones (see §10). Nothing was installed, built, run, pulled or modified.
- **Conventions:** paths are relative to the repo root of the section they appear in, or prefixed with the repo name. `mr/` abbreviates `who-said-what/app/src/main/java/com/meetingrecorder/`. **[unverified]** means not checked against an external source or not executed. **Plausible** means inferred from the code but not reproduced.

## Contents
1. Executive summary
2. Repos at a glance
3. Rule compliance
4. Assumption checks
5. Repo: `who-said-what` (Android)
6. Repo: `who-said-what-desktop` (Windows + NVIDIA)
7. Repo: `FalkorDB` ("Minutes", Team Vulcan)
8. Port map
9. Bugs, gaps and risks
10. Audit notes, corrections and open items

---

## 1. Executive summary

1. The team has three earlier repos. Together they cover audio → speakers → graph → cited Q&A, but no single repo does all of it.
2. `who-said-what` is a Kotlin/Compose phone app. It uses sherpa-onnx 1.13.8 (Silero VAD, Parakeet-TDT 0.6B v2 int8, TitaNet-small + ERes2Net) for enrollment-based speaker attribution, stores data in Room and answers "Ask" with BM25. It has no server, LLM or graph.
3. `who-said-what-desktop` is the Python port, built on a laptop identical to ours. It does streaming NeMo Sortformer diarization, per-speaker multi-talker ASR and TitaNet naming, with a FastAPI/WS/React UI, SQLite FTS5 and Qwen Q&A that cites `[mm:ss Name]`.
4. `FalkorDB` ("Minutes") is the graph half: a FalkorDB 6.0.1 company brain with bi-temporal facts, Gemma 4 + EmbeddingGemma, cited answers, SSE and a Next.js 16 UI. It has no audio, STT or diarization code, and it ingests diarized JSON in batches.
5. **RULE COMPLIANCE:** every commit in all three repos (2026-09-25 to 2026-10-12 IST) predates the 13-18 Oct window. All of it is prior art: disclose it, port or rewrite it inside the window, and never present it as new.
6. `FalkorDB` targeted Graph Hacks (Oct 15-18, 2026), which runs at the same time as our event. Its own README calls it pre-event work that needs an organiser ruling, so the double-submission risk must be settled first.
7. No repo has a LICENSE file. Android says "all rights reserved", desktop says MIT only in `pyproject.toml`, and FalkorDB says "MIT." only in its README. The owners must add a licence before public reuse.
8. Assumption checks: "Falcon DB" means FalkorDB (the only "Falcon" in the repos is Picovoice Falcon, a diarization SDK named in a survey); "dataison engine" means the diarization engine; all FalkorDB-hackathon features live in `reference\FalkorDB`; no "Retrieve" branding exists anywhere.
9. Best ports: desktop `live.py`, `pipeline.py`, `identify.py` and `compat.py` for the speaker side; FalkorDB `hiveboard.memory`, `cypher/`, the Gemma client, the answer agent and the provenance UI for the brain side; joined by a recorder-contract normaliser plus a Track→Person indirection.
10. Biggest gaps: no live per-utterance graph writer; no voice-first Q&A in any repo (no voice question input, no TTS); GPU-bound diarization that cannot run on Vercel, Netlify or Render as-is; live extraction at F1 0.49; unverified model IDs.

---

## 2. Repos at a glance

| Repo | What it is | Stack | Commit date range (IST; author = committer date) | Licence | Runs on Windows? |
|---|---|---|---|---|---|
| `who-said-what`<br>github.com/Hydra-Of-Malice/who-said-what | Native Android app "Meeting Recorder". Records a meeting on the phone, transcribes it and labels each line with speaker + timestamp. Fully on-device: no server, no cloud, no LLM. 37 tracked files, ~3.9k lines. | Kotlin + Jetpack Compose (BOM 2026.09.00), AGP 9.4.1, Gradle 9.6.0, Room 2.7.2, DataStore 1.1.7. sherpa-onnx 1.13.8 AAR with Silero VAD, Parakeet-TDT 0.6B v2 int8, TitaNet-small + 3D-Speaker ERes2Net. minSdk 26 / target 36 / compile 37, arm64-v8a. | **2026-09-25 22:15:09 → 2026-10-08 09:56:38** (3 commits, Aditya Arnav) | **None.** README: "No license has been granted: all rights reserved." | **Build: yes.** PowerShell model script + `gradlew.bat`; Android SDK present on this machine; build not attempted. **Run: no.** Android arm64 phone only (~6 GB RAM). |
| `who-said-what-desktop`<br>github.com/Hydra-Of-Malice/who-said-what-desktop | Local meeting transcriber (v0.1.1). Live overlap-aware diarization, one ASR stream per active speaker, TitaNet voiceprint naming, WebSocket live UI, SQLite FTS5, optional local LLM "ask the meeting" with `[mm:ss Name]` citations. | Python 3.12. torch 2.14.0 (cu126/cu130), NeMo pinned to git commit `cf724ac3…`, FastAPI 0.141.1, uvicorn 0.54.0, pywebview 6.2.1, PyAudioWPatch, soxr, silero-vad. Optional llama-cpp-python (Qwen3.5-4B GGUF). UI: React 19.3.0, Vite 8.3.1, Tailwind 4.3.3, TS 6.0.3. | **2026-09-26 11:28:06 → 2026-09-27 12:04:22** (40 commits, Aditya Arnav) | **None.** MIT is claimed only in `pyproject.toml:7` and the README. | **Yes, primary target:** native Win 10/11 x64, no WSL/Docker. **An NVIDIA GPU is mandatory (no CPU mode).** Its dev machine matches ours. Not installed or run in this audit. |
| `FalkorDB` ("Minutes", substrate "Hive Board", Team Vulcan)<br>github.com/Hydra-Of-Malice/FalkorDB | Company brain for *Graph Hacks: Context for AI Agents* (WeMakeDevs x FalkorDB, Track 3). Meeting transcripts and emails become Episodes, then bi-temporal Facts, Decisions and Commitments, then cited Q&A with Cypher and subgraph. Consumes diarized JSON produced elsewhere. | Python 3.12-3.13 uv workspace: FastAPI 0.142.2, falkordb 1.7.1, redis 8.1.0, mcp 2.2.0, onnxruntime 1.30.0, google-genai 2.28.0, openai 3.24.0. `falkordb/falkordb:6.0.1` (Docker). Next 16.3.8 + React 19.2.8 + Tailwind 4.3.3 + Base UI/shadcn + react-force-graph-2d + three. Gemma 4 via Gemini API / OpenRouter / Ollama; EmbeddingGemma-300m ONNX. | **2026-10-03 00:13:22 → 2026-10-12 09:55:46** (44 commits: Aditya Arnav 32, PDGamerSG 9, "Admin" 3) | **None.** Only the last line of README: "MIT." | **Yes, with Docker Desktop** for FalkorDB (daemon currently stopped). Python is Windows-clean (uvloop excluded on win32; ONNX on CPU). README commands are bash. Its README benchmark ran on this same i9-13900HX. |

---

## 3. Rule compliance

> **Bottom line:** none of the three repos contains code written inside the official development window. Treat them as **disclosed prior work**. Port or rewrite what we need **during 13-18 Oct 2026**, in a fresh public repo whose history starts in the window, and credit the sources.

### 3.1 What the handbook says (`docs/handbook/00-handbook.md`)
- `:186` The project development window is **13 to 18 October 2026**, including the on-site Hack Day.
- `:188` Teams may build before arriving at the venue, "but only within the stated development window".
- `:32` Build and demonstrate an **original** AI solution within that window.
- `:259` "Submit original work and accurately describe contributions." Artificial PR activity, misleading submissions or inaccurate participant details may lead to disqualification.
- `:258` Comply with the MLH Code of Conduct, **open-source licenses** and organiser instructions.
- `:187`, `:192-194` Submit a public GitHub repository, a **working live deployment URL**, and the selected track + problem statement.

### 3.2 Commit date range per repo (verified with `git log --date=iso`)

| Repo | Commits | First commit | Last commit | Authors | Last commit vs window start (13 Oct) | Status |
|---|---|---|---|---|---|---|
| `who-said-what` | 3 | `f40b063` 2026-09-25 22:15:09 +0530 (initial import of all engine code) | `b5c8807` 2026-10-08 09:56:38 +0530 ("Version 0.1.2: fix black screen, launch crash and dark/light mode") | Aditya Arnav (3) | ~5 days before | **Prior art** |
| `who-said-what-desktop` | 40 | `3ade835` 2026-09-26 11:28:06 +0530 ("Desktop app skeleton …") | `5bfb081` 2026-09-27 12:04:22 +0530 | Aditya Arnav (40) | ~16 days before | **Prior art** |
| `FalkorDB` | 44 | `bd5c030` 2026-10-03 00:13:22 +0530 ("Scaffold Minutes …") | `f9c068c` 2026-10-12 09:55:46 +0530 | Aditya Arnav (32), PDGamerSG (9), Admin (3) | ~1 day before | **Prior art, and tied to a concurrent event** |

Notes on the histories:
- **`who-said-what`.** The middle commit is `6590190` (2026-09-27, "Rewrite README"). The `b5c8807` message says its fixes were "committed locally on 2026-09-26 but never pushed".
- **`who-said-what-desktop`.** The first commit is already a full skeleton, so the history looks compressed.
- **`FalkorDB`, `README.md:5-7`.** It targets Graph Hacks, Oct 15-18, 2026. The status line (Oct 9) reads "local prototype, pre-event work", and the README says the repo must not be presented as work begun on Oct 15.
- **`FalkorDB`, `docs/track3-readiness.md` ("Eligibility comes first").** It asks for an organiser ruling on reuse.

### 3.3 What this implies
1. **Do not submit, fork or bulk-copy any of these repos as the hackathon project.** All of their code predates 13 Oct, so under `:188` none of it counts as in-window work.
2. **Reuse means a disclosed port or rewrite inside the window.**
   - New work goes in a new public repo whose first commit is dated on or after 13 Oct, with hackathon work kept clearly separate in git history.
   - Add a README section "Prior work and attribution". It should link the three repos, name the authors, and say for each component whether it was ported, rewritten or new.
   - This is what `:259` "accurately describe contributions" requires.
3. **Settle the Graph Hacks overlap first.** `FalkorDB` was built for an event running Oct 15-18, 2026, at the same time as ours. Submitting overlapping work to both events could disqualify the team from either.
   - Whether the team is still entering Graph Hacks is **[unverified]**.
   - The team needs a decision, and the repo's own docs ask for an organiser ruling.
4. **Credit every author.** `FalkorDB` has commits by PDGamerSG and "Admin" as well as Aditya Arnav. Code from people outside our 1-3 person team (`:256`) must be credited as theirs.
5. **Keep the attribution honest.**
   - Do not carry over `who-said-what-desktop/wsw/neutral.py`, which deliberately hides model and framework names.
   - If any part of the FalkorDB demo flow is reused, disclose that its polished demo used **gold-label replay and scripted answers** (`LLM_BACKEND=fake`; `docs/demo-script.md`).
6. **Licences.** All three repos lack a LICENSE file. The owners should add an OSI licence before any code is reused in our public repo (handbook `:258`; a Track 2 open-source entry depends on it). Model licences carry their own duties:
   - CC-BY-4.0 attribution for Parakeet-TDT and TitaNet-small.
   - A redistribution check for "NVIDIA Open Model" and OpenMDW-1.1 **[unverified]**.

---

## 4. Assumption checks

### 4.1 Is "Falcon DB" FalkorDB?
**Verdict: yes.** "Falcon DB" means FalkorDB. No product called "Falcon DB" exists in any repo or in our docs.

Evidence that the database is FalkorDB:
- The clone is `reference\FalkorDB`, remote `github.com/Hydra-Of-Malice/FalkorDB`.
- `docker-compose.yml` runs `falkordb/falkordb:6.0.1`, and the Python client is `falkordb` 1.7.1, connecting over a `falkor://` URL (`packages/hiveboard/hiveboard/db.py`).
- `README.md:5` targets "Graph Hacks: Context for AI Agents", the WeMakeDevs x FalkorDB event.

Search for "Falcon":
- Ran a case-insensitive `grep -ril falcon` over `reference/` and `docs/`. The only hits are in `who-said-what/RESEARCH.md`:
  - `:95` "Picovoice Eagle / Falcon": speaker recognition / diarization, an Apache-2.0 SDK that needs a **paid AccessKey validated online**.
  - `:250` the link `https://github.com/Picovoice/falcon`.
- So the only "Falcon" in the team's material is a **third-party diarization SDK** in a survey table, not a database. Nothing in any repo uses it.

### 4.2 Is the "dataison engine" the diarization engine? What is it in each repo?
**Verdict:** "dataison" is read as **"diarization"**. A case-insensitive search for `dataison|datason|dataisan|data-ison` over `reference/` and `docs/` returns **0 hits**, so this is an interpretation, not a literal match. What the "engine" is differs by repo:

| Repo | Diarization engine | What it actually is | Key files |
|---|---|---|---|
| `who-said-what` | **Enrollment-based speaker attribution**, not a generic diarizer. There is no pyannote and no sherpa-onnx `OfflineSpeakerDiarization`. | On sherpa-onnx 1.13.8:<br>1. Silero VAD.<br>2. Parakeet-TDT 0.6B v2 int8 offline transducer ASR (English) with token timestamps.<br>3. TitaNet-small + 3D-Speaker ERes2Net (VoxCeleb EN) ensemble embeddings on 2 s / 0.5 s-hop windows.<br>4. Centred, cohort-normalised matching against enrolled voices.<br>5. Viterbi turn labelling.<br>6. Provisional → 20 s block → final relabel passes.<br><br>Open set uses greedy centroid "Speaker N" clustering, which is reached only when no voices are saved (a bug, AND-B1). Spectral clustering is used only by the debug harness. Android arm64 only. | `mr/engine/SherpaEngine.kt`, `Pipeline.kt`, `TurnLabeler.kt`, `SpeakerMatcher.kt`, `Enrollment.kt`, `SpectralClustering.kt` |
| `who-said-what-desktop` | **True streaming diarization:** NVIDIA **Nemotron-3-Diarization (Sortformer)**, loaded as NeMo `SortformerEncLabelModel` (`wsw/engine/models.py:43-46, 232-240`). | Runs at 80 ms frames for up to 8 speakers.<br><br>Coupled through NeMo `SpeakerTaggedASR` with:<br>- `multitalker-parakeet-streaming-0.6b-v1` (one ASR stream per active track), or<br>- the optional multilingual `nemotron-3.5-asr-streaming-0.6b`.<br><br>Naming uses `speakerverification_en_titanet_large` (192-d). Offline Sortformer runs a final pass at stop.<br><br>NeMo is pinned to `cf724ac337d1ebc7d0dda1e23fb80916f52927a5` and torch to 2.14.0. Model IDs are as recorded in `wsw/engine/models.py` SPECS **[unverified against Hugging Face]**. It is a port of the Android app (`docs/ARCHITECTURE.md:3`). | `wsw/engine/live.py`, `pipeline.py`, `identify.py`, `embedder.py`, `finalize.py`, `offline.py`, `models.py`, `compat.py` |
| `FalkorDB` | **None.** | No STT, diarization or audio code. Input is a batch JSON "recorder export" said to be "Produced by the team's on-device recorder (speech-to-text plus speaker diarization)" (`docs/contracts/recorder-format.md:3`).<br><br>Diarization appears only as plans:<br>- `docs/06-implementation-plan.md:443`: Gemma 4 E4B audio input + pyannote, a stretch goal.<br>- `docs/review/02-ingest.md:65`: live audio upload is "Missing"; estimates are 4-6 h with cloud STT, or 12-16 h with faster-whisper + pyannote.<br><br>Searches for whisper, pyannote, diariz, sherpa, nemo, wav2vec, faster_whisper, deepgram, assemblyai, speechbrain, getUserMedia, MediaRecorder, SpeechRecognition, TTS and speechSynthesis hit only docs and docstrings. | `packages/ingest/ingest/recorder_format.py` |

### 4.3 Where the earlier FalkorDB-hackathon features live, and the full list (our passive features)
**Location:** all of them are in `reference\FalkorDB`. The product is "Minutes" by Team Vulcan, built for *Graph Hacks: Context for AI Agents* (WeMakeDevs x FalkorDB, Track 3 Company Brain).

| Area | Path |
|---|---|
| Queries | `cypher/`: 59 numbered `.cypher` files, verified with `ls` |
| Substrate | `packages/hiveboard/`: schema, bi-temporal memory, tenancy/ACL, CAS task board, skills, events, `hive.score` UDF, MCP server, bench |
| Ingest | `packages/ingest/`: recorder/email formats, chunker, EmbeddingGemma, Gemma 4 extraction v1-v5, eval |
| API | `apps/api/`: FastAPI, agents, SSE |
| Web | `apps/web/`: Next.js 16 UI |
| Data | `data/`: synthetic corpus + gold labels |
| Docs | `docs/`: contracts, pitch, bench, audit, `falkordb-issues.md` |

**Not in the who-said-what repos:**
- `who-said-what`: a case-insensitive search for `falkor|cypher|graph|neo4j|redis|gemma|ollama|llm|openai|gemini` hit only Compose `ui.graphics` imports and a `RESEARCH.md` mention of a third-party repo using Gemma 3 1B. Storage is Room (`mr/data/Db.kt`).
- `who-said-what-desktop`: a search for `falkor|cypher|neo4j|redis|graphdb|knowledge.graph` hit only "Redistributable" (`wsw/server/setup_flow.py:288`, `scripts/build-runtime.ps1:45,52`). Storage is SQLite WAL + FTS5.

**Caveat:** this work is not strictly earlier. Graph Hacks runs Oct 15-18, 2026, at the same time as our event; only the commits (Oct 3-12) are earlier. See §3.

#### A. Brain / back-end features
1. **Meeting import** of recorder-format JSON (`packages/ingest/ingest/recorder_format.py`, `pipeline.py run()`) and **email import** of `.json`, `.eml` and `.mbox` (`run_thread`).
2. **Speaker-attributed Episodes.** Consecutive same-speaker segments are merged into 30-60 s episodes (`chunker.py`). Writes create `SPOKEN_BY`, `ATTENDED`, `HAS_EPISODE` and `NEXT` edges (`cypher/10_ingest_episode.cypher`).
3. **LLM extraction of Facts, Decisions and Commitments** (v5). It uses 3-episode dialogue windows, a closed predicate vocabulary, an entity registry and a contradiction/refinement judge (`extract.py`, `prompts/extract_v5.md`, `vocab.py`, `dates.py`).
4. **Bi-temporal facts with supersession:**
   - `SUPERSEDES{at}` chains; `CONFIRMED_BY` for restatements; mid-chain insert for older statements.
   - Content-hash ids `fact_<ep>_<sha1[:8]>`.
   - Single-valued predicates: `due`, `owner`, `uses_vendor`, `uses_pattern`, `status`.
   - Sources: `cypher/11`, `12`, `18`, `19`, and `pipeline.py _store_item`.
5. **Belief at time T** (`cypher/31_belief_at_time.cypher`).
6. **Why / provenance path:** episode, speaker, container, `t_start`/`t_end` and the supersession chain (`cypher/32_why_path.cypher`).
7. **Template queries:**
   - who-knows: `cypher/50`
   - open commitments: `51`
   - contradictions: `52`
   - owners: `54`
   - stale decisions
   - WCC entity clusters
8. **Hybrid recall.** HNSW KNN plus a one-hop entity expansion, ranked by the `hive.score` JS UDF (recency × confidence × similarity), plus RRF over episode KNN and full-text results (`cypher/30_recall_hybrid.cypher`, `packages/hiveboard/hiveboard/udf/score.js`).
9. **Grounded answers with citations:**
   - A regex intent router chooses among a Cypher template, a graph traversal, vector search, or read-only free-form Cypher (3 s timeout, `LIMIT 200`).
   - Gemma cites `[fact_…]` / `[ep_…]` with speaker, meeting and timestamp. A missing or invented citation falls back to a grounded template answer.
   - Sources: `apps/api/agents/router.py`, `answer_agent.py`, `evidence.py`.
10. **Persisted chat sessions** as `Turn` and `ChatSession` nodes (`apps/api/hive_adapter.py:1339`).
11. **Live events.** Redis pub/sub on `team_<slug>:events` carries `episode.ingested`, `fact.created`, `fact.superseded`, `commitment.*` and `task.*`. They reach the browser as SSE with a replay buffer at `/tenants/{t}/events` (`hiveboard/events.py`, `EVENTS.md`, `apps/api/bus.py`, `apps/api/routes/board.py:261`).
12. **Multi-agent workers:** FactAgent, Scheduler, FollowUp, Standup and Answer. A date supersession wakes the Scheduler, which hands off to FollowUp (`apps/api/workers.py`, `apps/api/agents/*`).
13. **A2A task board:** compare-and-set claims on `version`, leases with a reaper, and handoffs that carry subgraph ids (`board.py`, `cypher/20`-`2B`).
14. **Learned Skills** with success rate and PageRank, plus Accept/Reject (`skills.py`, `cypher/40`-`46`).
15. **Decision impact ("ripple effect").** A `WorkItem` can `DEPENDS_ON` a Fact. When that fact is superseded, the item is flagged needs-review with its owner, and a local, never-sent follow-up draft is saved (`cypher/80`-`83`, `hiveboard/impact.py`).
16. **Multi-tenancy:**
   - One graph per tenant plus a `commons` graph.
   - rw and ro ACL users per tenant.
   - `GRAPH.COPY` snapshots; admin token; hashed per-tenant API keys.
   - Sources: `tenancy.py`, `apps/api/auth.py`.
17. **Embedding-space guard.** Each tenant's embedding model is recorded and mismatched imports are refused with 409/503 (`pipeline.check_embedding_model`, `apps/api/ingest_service.ensure_embedding_space`).
18. **MCP server** with tools recall, why, belief_at, list_tasks, claim, handoff and open_commitments (`packages/hiveboard/hiveboard/mcp_server.py`).
19. **Deterministic model-free modes:** `LLM_BACKEND=fake`, `EMBED_BACKEND=fake`, `HIVE_BACKEND=fake` (`.env.example`).
20. **Benchmark** of latency percentiles and CAS contention (`packages/hiveboard/bench/harness.py`, `docs/bench/`), and an **extraction eval** with strict and partial P/R/F1 (`packages/ingest/ingest/eval.py`).

#### B. User-facing features (`apps/web`, Next.js 16 App Router)

**Landing `/`:**
- "Your company, with a memory." hero with animated SVG threads (`components/landing/HeroFlow.tsx`).
- 24 s silent looping walkthrough video with VTT transcript and poster (`components/landing/CompanyWalkthrough.tsx`, `public/media/`).
- Source strip.
- "Know what changed" story (`components/landing/MemoryFeatures.tsx`).
- "Walk in with the context" questions (`components/landing/ContextQuestions.tsx`).
- Floating **Ask Minutes** assistant: pick a workspace, then continue in chat (`components/landing/LandingAssistant.tsx`).
- Three.js scroll-driven block sculpture with drag, WebGL fallback and reduced-motion support (`components/landing/LandingFinale.tsx`, `components/memory/MemoryScene.tsx`).

**Workspace shell** (`components/AppShell.tsx`):
- Sidebar with Overview, Sources, Ask Minutes, Agent board, Timeline, Skills, Workspaces and Bench.
- Breadcrumb and mobile sheet navigation.
- **Ctrl/Cmd+K** command menu that navigates or runs "Ask Minutes: <query>" (`components/CommandMenu.tsx`).
- Live API status: FalkorDB version, LLM/embedding info, and labels for "Scripted answers · demo mode", "Test embeddings" and "mock data".

**`/workspaces`** (`app/workspaces/page.tsx`): tenant directory with live counts and a create-workspace dialog.

**Overview `/t/[team]`** (`components/pages/CompanyOverview.tsx`):
- Ask box with prompt chips.
- Count strip.
- `DecisionCanvas` relationship view (`components/memory/DecisionCanvas.tsx`).
- **"The ripple effect"** decision-impact list:
  - Filters: needs review, aligned, closed, unresolved.
  - Baseline → current value and owner.
  - Original/Current decision evidence and a dependency-document dialog.
  - Force-graph path with an edge trail.
  - **Prepare follow-up:** a local, never-sent, copyable draft.
  - **Connect/Edit work** dialog (`components/impact/WorkItemDialog.tsx`).
  - Executed Cypher.
- "In the company memory": recent decisions.

**Sources** (`components/pages/SourcesPage.tsx`):
- Meeting import with progress, stats and errors; meeting table.
- **Transcript viewer:**
  - Per-speaker coloured chips and an episode list with mm:ss.
  - **Play at** seeks `<audio>`, or highlights the episode when no audio exists.
  - Follows playback; deep links `?meeting=&t=`.
- Email tab: `.json`/`.eml`/`.mbox` import, and a thread view with to/cc and highlighting.

**Ask Minutes chat** (`components/pages/ChatPage.tsx`):
- Sessions persisted in the URL, suggestions, `?q=` prefill.
- **"Why this answer"** panel:
  - Sources rendered as "Speaker said in Meeting at mm:ss [Play at]" or "Open email" (`components/SourceLink.tsx`).
  - Evidence episodes with scores.
  - Router decision.
  - Copyable Cypher.
  - Highlighted subgraph.
- Read-only Cypher typed into chat returns tables.

**Global overlays** (`components/Overlays.tsx`):
- "Behind the decision" Why sheet with the source quote, supersession history with validity ranges, Cypher and subgraph (`components/WhyPanel.tsx`).
- Email drawer.

**Board** (`components/pages/BoardPage.tsx`):
- Agent cards and open commitments.
- 5-column A2A kanban fed live by SSE.
- Event log and a task trace drawer with artifacts.
- Demo buttons "move it to Friday" and "Induce double claim".

**Timeline** (`components/pages/TimelinePage.tsx`): time slider comparing belief at time T with now, plus snapshots.

**Skills** (`components/pages/SkillsPage.tsx`): learned routines with success rate, PageRank, Accept/Reject and Run.

**Bench** (`app/bench/page.tsx`, `components/pages/BenchPage.tsx`): recharts latency percentiles and CAS contention.

**Graph view and Cypher display:**
- react-force-graph-2d with label colours and click-to-Why (`components/GraphView.tsx`, `graph/ForceGraphInner.tsx`).
- `CypherBlock.tsx`, `StateBadge.tsx`.

#### C. API-only features
Who-knows, contradictions, WCC clusters, stale decisions, ACL probe, `GRAPH.COPY` snapshots, admin token and per-tenant keys (`docs/contracts/http-api.md`). The only ACL-probe UI is in dead code, `components/pages/TenantsPage.tsx`.

### 4.4 "Retrieve" branding
**Verdict: not found.** No repo and no file in our `docs/` uses "Retrieve" as a product, brand or UI name. Every hit is a generic identifier or prose. If "Retrieve" is meant as our product name, it would be new.

| Where | Search | Result |
|---|---|---|
| `who-said-what` (all tracked files: code, `README.md`, `RESEARCH.md`, `docs/ARCHITECTURE.md`, `THIRD_PARTY_NOTICES.md`, `res/values`) | case-insensitive `Retriev` | **1 generic prose hit:** `RESEARCH.md:17` ("BM25 keyword retrieval"). The input audit reported 0 hits; corrected here (§10). Brand in use: "Who Said What" (README title), "Meeting Recorder" (`app/src/main/res/values/strings.xml` `app_name`), package `com.meetingrecorder`, `rootProject.name` "MeetingRecorder" (`settings.gradle.kts`). |
| `who-said-what-desktop` (all tracked files) | case-insensitive `retriev` | Generic only: `retrieve()` at `wsw/nlp/llm.py:450` (called at `:477`), "retrieval-augmented" in the `wsw/nlp/llm.py:15` docstring, and `docs/LLM.md:4,9,82,84`. Brand: "Who Said What" (`wsw/app.py:80` window title, README, `installer/WhoSaidWhat.iss` AppName). |
| `FalkorDB` (all tracked text except lockfiles, plus text extracted from `docs/pitch/minutes-pitch.pptx` and `docs/LinkedIn post — Graph Hacks announcement (draft v1).docx`) | case-insensitive `retriev` | 29 files, all generic. **Identifiers:**<br>- `cypher/40_skill_retrieve.cypher`<br>- `retrieve_skill`: `packages/hiveboard/hiveboard/skills.py:84,86`, `hiveboard/__init__.py:55,132`, `apps/api/hive_adapter.py:609,1522`, `apps/api/agents/tools.py:145,291,343`<br>- `AnswerAgent._retrieve` / `_retrieve_why`: `apps/api/agents/answer_agent.py:187,336,451`<br>- `standup_agent.py:4,66,123,124`<br>- `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` Gemini task types: `packages/ingest/ingest/embed.py:257`<br>- comment at `apps/web/lib/types.ts:209`<br>- tests<br>**Prose:** README, `docs/*`, `.21st/*`.<br>The pptx and docx text had 0 hits. PNG/JPG images and MP4 videos were not OCR'd or transcribed. Brand: "Minutes" (`apps/web/app/layout.tsx:16`, "minutes." wordmark in `AppShell.tsx`, package `@minutes/web`); substrate "Hive Board"; team "Team Vulcan". |
| All three repos | case-sensitive whole word `Retrieve` (`git grep -w`) | **0 hits** |
| `D:\HACKATHONS\hackoctober fest\docs\` (handbook + brief) | case-insensitive `retriev` | **0 hits** |

---

## 5. Repo: `who-said-what` (Android "Meeting Recorder")

Root: `D:\HACKATHONS\hackoctober fest\reference\who-said-what`. Remote `https://github.com/Hydra-Of-Malice/who-said-what.git`, branch `main`. 37 tracked files, ~3.9k lines; every source, build, resource and doc file was read.

### 5.1 What it is
A **native Android (Kotlin + Jetpack Compose) app** that records a meeting on the phone, transcribes it, and labels every line with a speaker name and a timestamp.
- **Everything runs on the device:** no server, no cloud API, no LLM.
- **Summaries** use extractive TF-IDF, and **"Ask"** uses BM25 keyword search (`README.md`, `docs/ARCHITECTURE.md`).
- **No graph, Gemma or web code.** It has nothing to do with FalkorDB.

### 5.2 Commit history
3 commits, dated 2026-09-25 to 2026-10-08 (§3.2). The author is Aditya Arnav, and author and committer dates match.

### 5.3 Stack and versions
There is no lockfile and no version catalog. Versions come from the Gradle scripts:
- **Build tooling:**
  - Gradle wrapper **9.6.0** (`gradle/wrapper/gradle-wrapper.properties`).
  - Daemon JVM toolchain **25** (`gradle/gradle-daemon-jvm.properties`).
  - Java source/target 17 (`app/build.gradle.kts`).
  - AGP **9.4.1**, Compose compiler plugin **2.2.10** (AGP built-in Kotlin 2.2.10), KSP **2.3.6** (`build.gradle.kts`).
- **Android targets:** `compileSdk 37`, `targetSdk 36`, `minSdk 26`, versionName 0.1.2. ABI is `arm64-v8a` only; `-PincludeX86=true` adds x86_64.
- **Libraries:** Compose BOM `2026.09.00`, Haze 1.7.3, activity-compose 1.10.1, navigation-compose 2.9.0, lifecycle 2.9.1, core-ktx 1.16.0, Room 2.7.2 (KSP), DataStore 1.1.7, coroutines 1.10.2.
- **sherpa-onnx 1.13.8:** `sherpa-onnx-static-link-onnxruntime-1.13.8.aar`.
  - `scripts/download-models.ps1` pulls it into `app/libs/`.
  - It is gitignored. Neither the AAR nor the models are in the clone.

### 5.4 Entry points and how to run

**Entry points:**
- **`App.kt`:** the Application class.
  - Loads models lazily (`engine()`).
  - When models load, deletes voice profiles whose embedding size differs (`App.kt:30`).
- **`ui/MainActivity.kt`:** the launcher.
  - NavHost routes are `home`, `meeting/{id}`, `setup`, `settings` and `speakers`.
  - Has a debug-only WAV harness (`MainActivity.kt:47-80`).
- **`service/RecordingService.kt`:** a foreground `microphone` service (`ACTION_START`/`ACTION_STOP`) that owns the mic, the WAV writer and the `Pipeline`.

**Run (README):**
1. `powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1` (~750 MB).
2. `.\gradlew.bat assembleDebug`.
3. `adb install app\build\outputs\apk\debug\app-debug.apk`.

You need an arm64 phone with ~6 GB RAM.

**Test without a mic:** `adb shell am start -n com.meetingrecorder/.ui.MainActivity --es debugWav <16k mono wav> [--ei enrolFromFile N] [--es participants "A,B"]`. This replays the file at ~4x real time (`RecordingService.kt:95-110`).

### 5.5 End-to-end pipeline

**Input.**
- `AudioRecord` uses the `VOICE_RECOGNITION` source at 16 kHz mono PCM16, delivered as 100 ms float chunks.
- Reader and dispatch threads are separate (`mr/audio/Audio.kt:19-58`).
- Every chunk goes to `WavWriter` and to `Pipeline.feed`.

**Models.** All run through sherpa-onnx (`mr/engine/SherpaEngine.kt:100-142`, `scripts/download-models.ps1`).
- **VAD:** `silero_vad.onnx`.
  - threshold 0.5, minSilence 0.25 s, minSpeech 0.25 s, window 512, maxSpeech 15 s.
  - Two instances: one live, one per slice.
- **ASR:** `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`.
  - English with punctuation, ~640 MB on disk, ~1 GB RAM.
  - Runs through `OfflineRecognizer` with `modelType="nemo_transducer"`, `greedy_search` and 80-dim fbank.
  - Returns text plus per-token timestamps.
- **Speaker embeddings:** `nemo_en_titanet_small.onnx` (`models/spk`) plus `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` (`models/spk2`).
  - Each vector is L2-normalised; the two are concatenated and scaled by 1/√2. One cosine then equals the mean of the two models' cosines (`SherpaEngine.kt:80-85`).
  - The embedding dimensions are not stated in the repo **[unverified]**.

**Mode: pseudo-streaming, not streaming ASR.**
1. **Slicing** (`mr/engine/Pipeline.kt:69-93`).
   - The live VAD counts silent 32 ms windows.
   - A slice is cut once the buffer reaches `sliceSeconds` (default 6 s, range 3-10) **and** at least 6 silent windows (~192 ms) have been heard.
   - A cut is forced at slice length + 3 s. Slices do not overlap.
2. **Pass 1, per slice, immediately** (`Pipeline.kt:105-130`).
   - The per-slice VAD finds speech spans; slices with less than 0.5 s of speech are skipped.
   - Parakeet decodes the whole slice.
   - Speech spans become **2 s windows with a 0.5 s hop**. Spans under 0.8 s are dropped; spans of 2.5 s or less are embedded whole (`TurnLabeler.kt:21-38`).
   - **Viterbi** runs over the participant scores with switch penalty 0.15 and a 1 s minimum turn. The penalty is multiplied by 0.15 across gaps of 1 s or more, and by 0.3 when adjacent windows have cosine below 0.65 (`TurnLabeler.kt:56-80`).
   - Tokens are assigned to turns by timestamp, and word pieces stay with their word (`Pipeline.kt:182-195`).
   - In closed-set mode, lines are written as `provisional`.
3. **Pass 2, per 20 s block** (`Pipeline.kt:200-213`).
   - Relabels with penalty 0.35 and a 1.5 s minimum turn, snapping boundaries to the nearest pause within 1 s.
   - Turns of 3 s or more that clear `speakerThreshold` with a margin of at least 0.06 adapt that person's profile (`SpeakerMatcher.adapt`, max 12).
4. **Pass 3, at stop** (`Pipeline.kt:216-219`): the same relabel over the whole meeting.

Each pass **deletes and re-inserts** the segments in its time range (`MeetingDao.deleteSegmentsInRange`, `Pipeline.kt:170`).

**Scoring** (`mr/engine/SpeakerMatcher.kt`).
- A running session mean (capped at 400) is subtracted from every vector.
- A turn's score against a participant is the mean of its top-3 centred cosines to that person's 2.5 s enrollment chunks.
- A "hub" term (the participant's mean similarity to the other participants) is subtracted.
- A line is marked `uncertain` when the margin to the runner-up is below 0.05.

**Open set ("Quick start").** Windows are clustered greedily against centroids into "Speaker N" groups using `clusterThreshold` 0.70 (`Pipeline.kt:158-164`).

**Enrollment gates** (`mr/engine/Enrollment.kt:14-29`). Each person says their name, then reads a fixed phrase, for up to 24 s. The sample must meet all of these:

| Gate | Requirement |
|---|---|
| Speech length | ≥ 10 s |
| SNR | ≥ 15 dB |
| Clipping | ≤ 0.5 % |
| Phrase words recognised | ≥ 60 % |
| Self-consistency | ≥ 0.60 |
| Separation from every other participant | ≥ 0.15 |

After 3 attempts, the best sample can be accepted anyway.

**Latency.** The repo has no measurements. From the code:
- A line can appear no sooner than one slice (6-9 s of audio) plus ASR and embedding time.
- Labels change again at ~20 s and at stop.
- Model load takes 5-10 s (README).
- `RESEARCH.md` quotes third-party phone RTFs for Parakeet 0.6B of 0.09-0.60 **[unverified]**.

**Output schema** (`mr/data/Db.kt`, Room/SQLite `meetings.db`, version 4):
- `Meeting(id, title, startedAt, endedAt, audioPath, participants CSV)`
- `Segment(id, meetingId, idx, startMs, endMs, speaker:String, text, confidence:Float, uncertain, embedding:FloatArray→LE float32 BLOB, provisional)`. `startMs` is relative to the start of the recording.
- `Speaker(name PK, embedding, createdAt, samplePath, chunks flattened)`

**Retrieval** (`mr/nlp/TextTools.kt`):
- TF-IDF top-k sentence summary.
- BM25 search (k1 1.5, b 0.75) over segments, returning the top 5 with speaker and time.
- A "my name is X" name extractor.

### 5.6 FalkorDB, licence, Windows
- **FalkorDB: none.** All storage is Room (see §4.3).
- **Licence:** there is no LICENSE file. `README.md:167` says "No license has been granted: all rights reserved." `THIRD_PARTY_NOTICES.md` lists:

  | Component | Licence |
  |---|---|
  | sherpa-onnx | Apache-2.0 |
  | ONNX Runtime | MIT |
  | Silero VAD | MIT |
  | Parakeet-TDT 0.6B v2 | CC-BY-4.0 |
  | TitaNet-small | CC-BY-4.0 |
  | 3D-Speaker ERes2Net | Apache-2.0 |

- **Windows build is the primary documented path.** The model script is PowerShell-only and uses `%SystemRoot%\System32\tar.exe -xjf`; `gradlew.bat` is present.
- **This machine** has an Android SDK at `%LOCALAPPDATA%\Android\Sdk` (platforms `android-36` and `android-37.0`, build-tools 36.0.0, `adb.exe`) and Java 25.0.1. `ANDROID_HOME` is not set. No build was attempted.
- **Runtime is Android arm64 only.** Nothing runs on Windows desktop or in a browser, and 4 GB emulators cannot hold the models (README).
- **Porting route:** the same ONNX models could be driven from sherpa-onnx's Python or other bindings on Windows. Whether the 1.13.8 wheel and CUDA work on this RTX 4060 is **[unverified]**.

### 5.7 Bugs and gaps
- **AND-B1: Quick start with saved voices never produces "Speaker N".**
  - With no participants, `matcher = SpeakerMatcher(enrolled)` loads *all* saved voices (`Pipeline.kt:40`, `RecordingService.kt:89-91`).
  - `labelWindows` then runs Viterbi over them (`Pipeline.kt:140-147`). `openMatch` is reached only when zero voices exist.
  - This contradicts the UI text (`SetupScreen.kt:210`) and the README.
- **AND-B2: No "unknown speaker" state in closed-set mode.**
  - Every speech window is forced onto a participant.
  - `speakerThreshold` only gates adaptation (`Pipeline.kt:208`), yet the Settings slider says "Higher = stricter matching" (`Screens.kt:382`).
- **AND-B3: `confidence` is not a cosine.**
  - It is a centred, hub-subtracted score (`SpeakerMatcher.kt:60`), although `Db.kt:42` documents it as a cosine.
  - In open set it is the constant 1f (`Pipeline.kt:153`).
  - It cannot be cited as a calibrated confidence.
- **AND-B4: Block and final passes delete and re-insert rows.** Segment ids change, and a rename made during the live meeting is overwritten at stop.
- **AND-B5: Short interjections get someone else's label.**
  - Speech under 0.8 s gets no embedding.
  - A slice with no turns inherits `lastLabel` (`Pipeline.kt:168`).
  - Overlapping speech is not handled.
- **AND-B6: WAV parsing assumes a 44-byte header** with 16 kHz mono and does no resampling (`Audio.kt:93-103`, `RecordingService.kt:98`, `MainActivity.kt:57`). The header is patched only on a clean stop.
- **AND-B7: `RecordingService` can leave a stale foreground service.**
  - It calls `startForeground` before `begin()`, and `begin()` returns silently when the state is not Idle (`RecordingService.kt:62-74`).
  - The setup screen's "Quick start" button is not state-gated (`SetupScreen.kt:216`).
- **AND-B8: Any schema bump wipes all data.** The database uses `fallbackToDestructiveMigration(true)` with `exportSchema=false` (`Db.kt:125`).
- **AND-B9: Long meetings may be slow and memory-heavy (not measured).**
  - All windows and tokens stay in memory for the final pass.
  - `absorbShort` costs O(turns×windows) per iteration (`TurnLabeler.kt:101-117`).
  - `raw()` re-centres every enrolled chunk on every call.
- **AND-B10: No tests, no export, English only.** Thresholds were tuned only on sherpa-onnx's two-speaker test file (`docs/ARCHITECTURE.md`).

### 5.8 Graph mapping suggested by this repo
The algorithmic core is pure math and ports cleanly to Python/numpy. `Segment` maps directly onto the graph as `(:Speaker)-[:SAID]->(:Utterance {startMs,endMs,text,confidence,uncertain,provisional})-[:IN]->(:Meeting)`. Labels are revised after the fact (AND-B4), so live graph writes need an upsert or revision mechanism.

---

## 6. Repo: `who-said-what-desktop`

Root: `D:\HACKATHONS\hackoctober fest\reference\who-said-what-desktop`.
- Remote: `github.com/Hydra-Of-Malice/who-said-what-desktop`.
- HEAD: `5bfb081`.
- Version: 0.1.1 (`wsw/__init__.py`, `pyproject.toml`).

### 6.1 What it is
- **Purpose:** a fully local meeting transcriber that runs natively on Windows and needs an NVIDIA GPU.
- **Diarization:** live and overlap-aware, using **NVIDIA Nemotron-3-Diarization (Sortformer)**, with **one multi-talker ASR stream per active speaker**.
- **Naming:** **TitaNet** voiceprints turn anonymous tracks into names.
- **Output:** utterances carry ms timestamps and stream over WebSocket into **SQLite FTS5**.
- **Q&A:** an "ask the meeting" RAG cites `[mm:ss Name]` (`wsw/nlp/llm.py`). Its backend is Qwen3.5-4B on llama.cpp. Gemma and Ollama never appear.
- **Graph:** no FalkorDB or graph code. The natural graph-write hooks are `MeetingSession._upsert`, `_relabel` and `_finalize` in `wsw/engine/pipeline.py`.
- **Hardware:** the dev machine in `docs/ARCHITECTURE.md` "Results" is **identical to ours**: i9-13900HX, RTX 4060 Laptop 8 GB, Windows 11.

### 6.2 Commit history
40 commits by one author, 2026-09-26 11:28:06 to 2026-09-27 12:04:22 +0530 (§3.2). The first commit is already a full skeleton.

### 6.3 Stack and versions
All versions come from the repo's lockfiles; their existence upstream was not checked.

| Layer | Pins (source) |
|---|---|
| Python | `requires-python >=3.12,<3.13` (`pyproject.toml`) |
| ML | `torch==2.14.0` + torchaudio from `download.pytorch.org/whl/cu126` or `cu130`. Not in the lockfile; pinned in `wsw/launcher.py:43`, `scripts/setup.ps1:38` and `wsw/server/setup_flow.py:45` |
| NeMo | `nemo-toolkit @ …/NVIDIA-NeMo/Speech/archive/cf724ac337d1….zip` (`requirements.lock:71`). The docs say the 3.0.0 wheel cannot load the diarizer |
| Server | fastapi 0.141.1, uvicorn 0.54.0, starlette 1.7.0, pydantic 2.13.5, pywebview 6.2.1 (`requirements.lock`) |
| Audio | pyaudiowpatch 0.2.12.8, soxr 1.1.0, soundfile 0.14.0, silero-vad 6.2.3 |
| Other | numpy 2.5.3, scipy 1.18.1, huggingface-hub 1.33.0, transformers 5.17.0, lightning 2.4.0, lhotse 2.0.0a6, pyannote-metrics 4.1 |
| Optional LLM | `llama-cpp-python >=0.3.33,<0.4` (`scripts/download-llm.ps1:12`); the 0.3.35 cu125 wheel was measured |
| UI | React/react-dom 19.3.0, Vite 8.3.1, Tailwind 4.3.3, TypeScript 6.0.3, lucide-react 1.48.0, oxlint 1.85.0 (`ui/package-lock.json`) |
| Packaging | python-build-standalone `3.12.14+20260924` (`scripts/build-runtime.ps1:5`), Inno Setup ≥6.3 |

### 6.4 Entry points and how to run
- **Setup:** `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-SkipModels] [-SkipUi] [-Cuda auto|cu126|cu130]`.
  - Creates a uv venv and installs torch and the lockfile.
  - Downloads the 3 required models (~2.8 GB).
  - Runs `npm run build`.
- **App:** `.venv\Scripts\python.exe -m wsw.cli run [--dev] [--browser] [--headless] [--port N]` (`wsw/cli.py`, `wsw/app.py`).
  - Uvicorn binds `127.0.0.1:<free port>` with a per-launch token (`?t=` or `X-Token`).
  - A WebView2 window opens on top.
- **Batch CLIs:**
  - `wsw transcribe FILE --engine multitalker|multilingual -v` prints step timings and writes no DB.
  - `wsw process FILE --participants a,b` runs the full pipeline into the DB.
  - `wsw devices`.
  - `wsw models --download diarizer,multitalker,embedder`.
- **UI dev:** `cd ui; npm run dev` (port 5173), then `wsw.cli run --dev`.
- **Tests:** `pytest tests -q` runs 141 test functions; models, mic and GPU are stubbed.
- **GPU evals:** `scripts/synth_meeting.py` and `scripts/eval_finalize.py` need `testdata/LibriSpeech/dev-clean`, which is gitignored.
- **Env vars:** `WSW_DATA_DIR`, `WSW_MODELS_DIR`, `WSW_TOKEN`, `WSW_NO_WARM`, `WSW_NO_SETUP`, `WSW_LLM_MODEL|PATH|DEVICE|IDLE_SEC`.

### 6.5 End-to-end pipeline
1. **Capture** (`wsw/audio/capture.py`).
   - PyAudioWPatch opens the WASAPI mic and, optionally, the default-output loopback at their native rates.
   - Each side is downmixed and resampled to **16 kHz mono float32** with soxr, aligned by sample count, and emitted as **100 ms blocks**.
   - A `Ducker` cuts the mic by ~22 dB whenever system audio is active. This is echo handling without AEC.
   - The recording is a 16-bit, 16 kHz WAV with mic on L and system on R.
   - File input goes through `read_wav_mono`, which accepts any WAV/FLAC and resamples.
2. **Live step** (`wsw/engine/live.py`). This is **true streaming**: one coupled step per **1.12 s** hop.
   - NeMo `SpeakerTaggedASR` runs a diarizer step that yields cumulative activity **(T, 8) at 80 ms**.
   - Each active track then gets its own ASR stream. The `multitalker` engine is speaker-kernel conditioned; the `multilingual` engine masks other tracks and takes a locale prompt.
   - Attention context is `[70,13]` or `[56,13]`, with bf16 autocast when the GPU supports it.
   - Sentences come from NeMo `seglsts` as `{speaker, start_time, end_time, words}`, and the session diffs them each step.
3. **Naming** (`wsw/engine/identify.py`), every **4 s**.
   - Only solo-speech frames are used: 2 s windows with a 0.5 s hop, embedded by TitaNet (192-d, `wsw/engine/embedder.py`).
   - `SpeakerMatcher` scores each window with a mean-centred cosine over the top-3 chunks, minus a cohort term.
   - Tracks are assigned to people one-to-one with the Hungarian algorithm (threshold 0.30, ambiguity margin 0.05, `wsw/config.py`).
   - Unknown tracks become `Speaker N` and can be named from a self-introduction (`wsw/nlp/text.py:extract_name`).
   - Names are re-solved retroactively, and stored lines are relabelled.
4. **Persist and emit** (`pipeline.py:210-258`).
   - Segments are upserted by key `(track, round(start, 2))` and flagged `provisional`.
   - They also get an `overlap` flag when more than 25 % of frames have another track active.
   - Events on `/ws`: `segment`, `activity`, `relabel`, `status`, `models`, `enroll`, `setup`, `llm`, `summary`, `renamed`.
5. **Final pass at stop** (`wsw/engine/finalize.py`, `offline.py`).
   - Offline Sortformer runs over the whole file (the "30.4 s profile").
   - Live tracks are aligned to offline tracks with Hungarian matching on a 10 ms grid, and identities are re-solved.
   - Segment starts are snapped to true onsets (by at most 1.5 s), and overlap flags are recomputed (threshold 0.15).
   - `provisional` is set to 0. **Text is never re-transcribed.**
6. **Q&A** (`wsw/nlp/llm.py`).
   - Retrieval is BM25 top-12 plus ±1 neighbouring lines, padded to at least 8 lines.
   - The prompt requires `[mm:ss Name]` citations, and `_CITE` flags the cited sources.
   - Long meetings are summarised map-reduce style; the context window is 8192 tokens.
   - Retrieval is lexical and covers a single meeting only (`docs/LLM.md:84`).

**Segment schema** (`wsw/store/db.py`):
- `segments(id, meeting_id, start_ms, end_ms, track 0..7, speaker, text, confidence, uncertain, overlap, provisional, words JSON)`.
- Related tables: `meetings`, `speakers` (mean and chunk embeddings as float32 blobs) and `meeting_tracks`.
- Storage is SQLite WAL + FTS5, with a `VACUUM INTO` backup on every open (`wsw/store/db.py:106-130`).
- Exports: MD, SRT and JSON (`wsw/server/api.py:362-372`).

**Latency** (author-measured in `docs/ARCHITECTURE.md` "Results", 2026-09-26; not re-run here):

| Workload | Time per 1.12 s step | Peak VRAM |
|---|---|---|
| 4 ASR streams | 182 ms (p95 233 ms) | 3.2 GB |
| 8 streams, live capture | 206 ms | 3.4 GB |
| Multitalker engine, 8 streams | 310 ms | n/a |
| Multilingual engine | 501 ms | n/a |

Other author-reported figures:
- Offline diarization: 21 min of audio in 6.5 s, using 1.5 GB VRAM.
- First NeMo import takes ~4 min; later imports ~8 s.
- Lines appear ~1 s after they are spoken.
- Attribution on a synthetic 4-voice meeting: 97.9 % solo, 97.7 % overlapped.
- Optional 4B Q4 LLM: adds ~3.4 GB VRAM.

### 6.6 FalkorDB, licence, Windows
- **FalkorDB: none** (§4.3).
  - Design note for the port: names change after the fact (`_relabel`, `refine`). Model the graph as `(:Utterance)-[:ON_TRACK]->(:Track)-[:RESOLVED_AS]->(:Person)`, so a relabel is a single edge update rather than a rewrite of every utterance.
- **Licence:** there is no LICENSE file (`git ls-files`).
  - `pyproject.toml:7` declares `license = { text = "MIT" }`, and the README "License" section says "MIT for the app code (see `pyproject.toml`)".
  - The installer bundles `..\LICENSE` only if the file exists (`installer/WhoSaidWhat.iss:91-93,108`).
  - Model licences per the README: OpenMDW-1.1, "NVIDIA Open Model", CC-BY-4.0, and Apache-2.0 for the optional GGUF.
- **Windows:** native Win 10/11 x64 is the main target, with no WSL or Docker.
  - On win32, `wsw/engine/compat.py:37-42` swaps NeMo's `torch.compile`-d flex-attention for the eager kernel, because triton has no Windows build.
  - Windows-only pieces: WASAPI loopback (PyAudioWPatch), `winreg`, WebView2, PowerShell scripts and Inno Setup.
  - **An NVIDIA GPU is mandatory; there is no CPU mode** (README "Known limits").
  - CUDA selection: cu126 needs driver ≥525; cu130 needs driver ≥580 and is the default for Turing and newer (`setup_flow.py:172-186`).
  - It fits our machine (Python 3.12 installed, RTX 4060 8 GB, no Docker needed).

### 6.7 Bugs and gaps
- **DSK-B1: Self-intro aliasing creates false names.**
  - `extract_name(..., strong_only=True)` runs on every unnamed track's speech (`pipeline.py:212-220`, `text.py:67,106`).
  - In a scratch-copy test, "I'm going to check" became `Going`, "I'm sure" became `Sure` and "I am happy" became `Happy`.
  - The aliases are never cleared. In a knowledge graph they would become bogus Person nodes.
- **DSK-B2: Reprocess and import are invisible while running.**
  - `state.session = MeetingSession.from_file(...)` is assigned only after the whole replay finishes (`api.py:399-401,419-420`; `pipeline.py:133-152`).
  - `/api/status`, the 409 busy guards and `llm_routes._meeting_busy` therefore miss it, so concurrent GPU jobs can start.
- **DSK-B3: Unauthenticated SPA catch-all.**
  - `@app.get("/{path:path}")` has no auth dependency and returns `FileResponse(dist / path)` (`api.py:509-514`).
  - On Windows, `Path / "C:/…"` resolves to the absolute path (pathlib behaviour verified), so a local process may be able to read arbitrary files.
  - Plausible; not tested against a running server.
- **DSK-B4: Unconditional bf16 offline.** `diarize_offline` uses bf16 autocast and ignores `resolve_precision` (`offline.py:71`). This is a risk on pre-Ampere GPUs **[unverified]**.
- **DSK-B5: `_identify` re-concatenates the whole meeting audio every 4 s** (`pipeline.py:189-190,242`), and all audio stays in RAM. Cost grows with meeting length; plausible slowdown.
- **DSK-B6: Upsert by `(track, start)` can leave stale rows.** If NeMo moves a sentence start, the old row could remain [plausible, depends on NeMo].
- **DSK-B7:** `scripts/llm_check.py:75` reads `st["model"]["path"]`, which `status()` no longer returns, so it raises `KeyError`.
- **DSK-B8:** `scripts/audio_check.py:297` expects mix = mic + system, but the `Ducker` (`capture.py:553`) changes the mix.
- **DSK-B9: Docs have drifted from the code.**
  - `ARCHITECTURE.md:77` says refine is "not yet called", but `pipeline.py:271` calls it.
  - The docs say the offline ASR is unloaded at meeting start, but commit `6157997` keeps it resident.
  - `RESEARCH.md:112` gives a threshold of 0.55, but `config.py:74` uses 0.30.
  - The status schema in `LLM.md:90` is stale.
- **DSK-B10: Unvalidated settings.** `PUT /api/settings` accepts unvalidated JSON (`api.py:193-209`). `final_pass` and `id_hop_sec` have no UI control.
- **DSK-B11: `wsw/neutral.py` hides model and framework names** in logs, errors and UI. This works against an open-source, verifiable entry.
- **DSK-B12: Unchecked downloads.** The launcher downloads uv from `releases/latest` without a checksum (`launcher.py:51`, `launch.ps1:42`). Only `build-installer.ps1` verifies the SHA-256.

**Skip when porting:** the installer, launcher, setup flow and `neutral.py`.

---

## 7. Repo: `FalkorDB` ("Minutes", Team Vulcan)

Root: `D:\HACKATHONS\hackoctober fest\reference\FalkorDB`.
- Remote: `github.com/Hydra-Of-Malice/FalkorDB`.
- Substrate package: "Hive Board".
- Target: Graph Hacks: Context for AI Agents (Oct 15-18, 2026), Track 3 Company Brain.

### 7.1 What it is
A company brain on FalkorDB.
- **Input:** recorder-format meeting transcripts and email threads.
- **Processing:** inputs become Episodes, then Facts, Decisions and Commitments with bi-temporal supersession, in one graph per team.
- **Output:** answers with citations, Cypher and a subgraph (`README.md`, `PRODUCT.md`).
- **Not included:** audio, STT and diarization. It consumes diarized JSON produced elsewhere (`docs/contracts/recorder-format.md:3`).
- **Status:** it describes itself as a "local prototype, pre-event work" (`README.md:7`).

### 7.2 Commit history
44 commits on `main`, 2026-10-03 00:13:22 to 2026-10-12 09:55:46 +0530. Authors: Aditya Arnav (32), PDGamerSG (9), "Admin" (3). See §3.

### 7.3 Stack and versions (from `uv.lock`, `pyproject.toml`, `apps/web/package.json`, `pnpm-lock.yaml`)
- **Python:** `requires-python >=3.12,<3.14`. The uv workspace holds `packages/hiveboard`, `packages/ingest` and `apps/api`.
  - fastapi 0.142.2, uvicorn 0.54.0, falkordb 1.7.1, redis 8.1.0, pydantic 2.13.5, sse-starlette 3.5.0, mcp 2.2.0.
  - onnxruntime 1.30.0, tokenizers 0.23.2, huggingface-hub 1.33.0, numpy 2.5.3.
  - openai 3.24.0, google-genai 2.28.0, httpx 0.28.1, pytest 9.1.1.
- **DB:** `falkordb/falkordb:6.0.1` with `MAX_QUEUED_QUERIES 500` (`docker-compose.yml`).
- **Web:** pnpm 10.33.4, lockfile v9.
  - Framework: next 16.3.8 (App Router, standalone output), react 19.2.8.
  - UI kit: @base-ui/react 1.8.0, shadcn "base-nova", tailwindcss 4.3.3.
  - Visuals: three 0.186.1, react-force-graph-2d 1.29.2, recharts 3.10.1, lucide-react 1.50.0, sonner 2.0.8.
  - typescript 5.9.3.
- **Models, as declared in the repo** (none checked against providers **[unverified]**):
  - LLM default `gemma4:e4b` on Ollama (`apps/api/config.py:35`); `gemma4:12b` is the alternative.
  - `gemma-4-26b-a4b-it` on the Gemini API with `thinking_level="minimal"`. The repo says it was "verified 2026-10-03" (`apps/api/llm/gemma.py:370`).
  - `google/gemma-4-26b-a4b-it` on OpenRouter (`docs/contracts/llm.md`).
  - Embeddings: `onnx-community/embeddinggemma-300m-ONNX`, 768-d, L2-normalised, with the model card's query/document prefixes, on the **CPU** execution provider. `gemini-embedding-001` (768-d) is the fallback, plus a hashed fake (`packages/ingest/ingest/embed.py:30,138`).

### 7.4 Entry points
- **API:** `uv run --package minutes-api uvicorn apps.api.main:app --port 8080`, run from the repo root because `apps` is a namespace package. Swagger is at `/docs`.
- **Seed:** `data/seed.py --tenants orbit,nimbus,quill --extract gold|real`, then `data/seed_work.py --tenants orbit`.
- **Web:** `cd apps/web && pnpm dev` on port 3001 (port 3000 is FalkorDB Browser).
- **MCP:** `hiveboard-mcp` / `python -m hiveboard.mcp_server` over stdio.
- **Eval:** `python -m ingest.eval`.
- **Bench:** `python -m bench.harness`.
- **Model-free modes:** `LLM_BACKEND=fake EMBED_BACKEND=fake`, where `--extract gold` replays gold labels. `HIVE_BACKEND=fake` gives a fully in-memory API (`.env.example`).

### 7.5 Graph schema (`cypher/00_schema.cypher`, `packages/hiveboard/hiveboard/schema.py`)
Each tenant has its own graph, `team_<slug>`. A shared `commons` graph holds the tenant registry, snapshots and published skills. All times are epoch ms.

**Labels and key properties:**

| Label | Key properties |
|---|---|
| `Person` | id, name, role, email |
| `Meeting` | id, title, started_at, ended_at, source_uri |
| `EmailThread` | |
| `Episode` | id, seq, source_kind, text, t_start, t_end, t_occurred, t_ingested, message_id, embedding |
| `Entity` | id, name, name_key, type, embedding |
| `Fact`, plus label `Decision` or `Commitment` | id, kind, subject, predicate, object, key=`subject\|predicate`, text, confidence, status (active/superseded/retracted), valid_from, valid_to, created_at, expired_at, due_at |
| `Turn`, `ChatSession` | Created inline in `apps/api/hive_adapter.py:1339`, with **no index** |

Other labels: `Task`, `Agent`, `Capability`, `Skill`, `Step`, `Artifact`, `Snapshot`, `Tenant`, `WorkItem`, `Document`, `FollowUpDraft`.

**Relationships:**

| Group | Relationships |
|---|---|
| Episode structure | `HAS_EPISODE`, `NEXT`, `SPOKEN_BY`, `ATTENDED`, `SENT_BY`, `SENT_TO{cc}`, `REPLIES_TO`, `MENTIONS` |
| Entity co-occurrence | `CO_MENTIONED{weight}` |
| Fact provenance | `DERIVED_FROM`, `STATED_BY`, `MADE_BY`, `OWNED_BY`, `DECIDED_IN`, `ABOUT`, `SUPERSEDES{at}`, `CONFIRMED_BY`, `PRECEDENT_FOR` |
| Task board | `CLAIMED_BY`, `HANDED_OFF`, `REQUIRES`, `USED`, `PRODUCED`, `LEASE_EXPIRED` |
| Skills | `STEP`, `DEPRECATED_BY` |

**Indexes:**
- About 36 range indexes.
- Full-text on `Episode.text`, `Fact.text` and `Decision.text`.
- HNSW vectors (768-d, cosine) on Episode, Fact, Skill and Entity.
- A unique `id` constraint on 17 labels via `GRAPH.CONSTRAINT`, with FAILED-constraint recovery.

**Queries:** 59 numbered files in `cypher/NN_*.cypher`, loaded by name (`hiveboard/queries.py`). `RealHive` (`apps/api/hive_adapter.py`) still holds about 40 inline queries.

**FalkorDB specifics:**
- The Python client uses `falkor://` (`hiveboard/db.py`); the connection is lazy, with timeouts.
- Each tenant gets its own Redis ACL users.
- The `hive.score` UDF is loaded with `GRAPH.UDF LOAD` and auto-loads on the first missing-function error (`udf/loader.py`).
- **Known 6.0.1 bug:** a `MATCH…WHERE` followed directly by another `MATCH` silently drops the first filter (`docs/falkordb-issues.md`). New Cypher needs `WITH` barriers.

### 7.6 Ingestion pipeline (`packages/ingest/ingest/pipeline.py`)
**Input contract** (`packages/ingest/ingest/recorder_format.py:25,85-156`; `docs/contracts/recorder-format.md`):
```
{version:1, meeting{id,title,started_at(tz-aware),ended_at,audio_uri}, speakers[{id,name,role}],
 segments[{start_ms,end_ms,speaker_id,text}]}
```
- Validation is strict: `extra="forbid"`, and speakers must be known.
- Segments must be time-ordered and **non-overlapping**; overlaps are rejected in `_consistent`.
- Blank and zero-length segments are dropped.

**Import:** `POST /tenants/{t}/meetings/import` returns 202 with a job id.
- The job runs in a bounded **in-memory** store of 500, which is lost on restart.
- The UI polls the job every second (`docs/contracts/http-api.md`, `SourcesPage.tsx` ImportPanel, `apps/api/jobs.py`, `apps/api/routes/meetings.py`).
- Processing is **batch per meeting, not streaming.**

**`run()` steps:**
1. **Validate** the input.
2. **Chunk.** Consecutive same-speaker segments become 30-60 s episodes with ids `ep_<container>_<seq>`. Segments are never split, and `t_start`/`t_end`/`t_occurred` plus segment indices are kept (`chunker.py`).
3. **Embedding guard and source-changed check**, then upsert the Meeting and Person nodes.
4. **Embed** each episode with the prefix `"title: none | text: …"`.
5. **Write** with `remember_episodes` / `cypher/10_ingest_episode.cypher`, then re-link `NEXT` (`pipeline.py:377-453`).
6. **Extract** with the v5 LLM extractor:
   - 3-episode dialogue windows and a closed predicate vocabulary (`vocab.py`).
   - An entity registry and a judge (`extract.py`, `prompts/extract_v5.md`).
   - Gold replay is the alternative.
7. **Store** each item through `_store_item`.

**How `_store_item` resolves facts:**
- Only `due`, `owner`, `uses_vendor`, `uses_pattern` and `status` are single-valued.
- A restatement becomes `CONFIRMED_BY`.
- A contradiction, decided by the LLM judge or a hint, calls `supersede_fact`.
- A statement older than the current head calls `insert_fact_in_chain`.
- Fact ids look like `fact_<episode_id>_<sha1[:8]>`. Each fact write is one atomic Cypher statement (`cypher/18_remember_fact`, `12_supersede_fact`, `19_insert_fact_in_chain`).

**Emails** go through `run_thread`.

**Events:** published on Redis channel `team_<slug>:events` (`hiveboard/EVENTS.md`). A date supersession wakes the Scheduler, which hands off to FollowUp.

### 7.7 Retrieval and Q&A (`apps/api/agents/answer_agent.py`, `router.py`, `evidence.py`, `prompts/answer.md`, `routes/chat.py`)
1. **Route.** A regex router chooses one of:
   - a numbered Cypher template: `why`, `who_knows`, `owners`/ownership, `open_commitments`, `belief_at`, `contradictions`, `stale`, `clusters`
   - a graph lookup
   - vector recall
   - read-only free-form Cypher (3 s timeout, `LIMIT 200`)

   Routing code: `router.py:64-245`.
2. **Hybrid recall** (`cypher/30_recall_hybrid`): KNN, one-hop expansion through shared entities, then ranking with `hive.score` (recency × confidence × similarity).
3. **Episode recall:** KNN and full-text results are fused with reciprocal-rank fusion (RRF).
4. **Dossiers:** each candidate fact gets one containing its full `SUPERSEDES` chain.
5. **Answer:** Gemma cites `[fact_…]`/`[ep_…]`, and each source carries speaker, meeting and `t_start`. A missing or invented citation falls back to a grounded template answer.
6. **Provenance:** `why()` (`cypher/32_why_path`) returns the episode, speaker, container and chain.

**There is no voice input and no TTS.**

### 7.8 Latency, evaluation, tests
**Latency (repo-stated):**

| Operation | Time |
|---|---|
| Graph recall | 4 ms |
| why | 1 ms |
| claim | 2 ms |
| remember + supersede | 3.5 ms |
| Single queries generally (README) | ~1-4 ms |
| Gemini call, minimal thinking | ~2 s |
| Chat answers (free tier; `docs/overnight-report.md`, deck slide 11) | 6-17 s |

- The Gemini free tier is capped at **16k input tokens per minute** (repo claim).
- Embedding and Ollama latency are unmeasured **[unverified]**.

**Gold labels and corpus:**
- `data/gold/*.json`: 17 files with 129 items; fields are kind, subject, predicate, object, speaker_id and segment_index.
- The corpus is synthetic: 10 meetings and 7 email threads across 3 fictional tenants (orbit, nimbus, quill) with planted supersessions.
- Segment timings are computed from word counts (`data/generator/gen_lib.py`). **No audio** (`*.wav` is gitignored).

**v5 extraction results** on 108 items (`docs/eval/extraction.md`):

| Scoring | Precision | Recall | F1 |
|---|---|---|---|
| Strict | 0.42 | 0.59 | 0.49 |
| Partial credit | 0.49 | 0.69 | n/a |

- The run took 261 calls over ~65 min, paced to the free tier.
- **Leakage:** the subject vocabulary in `vocab.py` was written from the gold labels.
- The polished demo replays gold labels and uses scripted answers (`docs/demo-script.md`).

**Tests:**
- 367 `def test_` functions: hiveboard 84 (need a live FalkorDB, otherwise skipped), ingest 111, API 172 (FakeHive, FakeLLM, FakeEmbedder).
- Recorded runs say "234 passed" (`docs/audit/pytest-impact-editor-2026-10-10.txt`; `docs/track3-readiness.md` also says 234). `docs/review/README.md` reports 86/111/172.
- Not re-run here, because that needs installs and Docker.

### 7.9 Multi-tenancy, task board, MCP
- **Tenancy** (`tenancy.py`, `apps/api/auth.py`):
  - One graph per tenant plus rw and ro ACL users, with selectors that make `commons` read-only.
  - `GRAPH.COPY` snapshots.
  - Admin token and hashed per-tenant API keys.
- **Board** (`board.py`, `cypher/20`-`2B`):
  - A2A task states, CAS claims on `version`, leases with a reaper, and handoffs carrying subgraph ids.
  - Workers: FactAgent, Scheduler, FollowUp, Standup and Answer (`apps/api/workers.py`).
  - Skills are ranked with PageRank (`skills.py`); WCC clusters come from `analytics.py`.
- **MCP** (`packages/hiveboard/hiveboard/mcp_server.py`): FastMCP tools recall, why, belief_at, list_tasks, claim, handoff and open_commitments. It uses the fake embedder unless a real one is injected.

### 7.10 UI, design language, deployment, pitch material
- **Features:** see §4.3 B.
- **Look as implemented:**
  - Base UI and shadcn primitives on Tailwind v4 tokens (`app/globals.css`).
  - Palette: paper `#f8faff`, navy `#182138`, cobalt `#3155ed`.
  - Type: self-hosted Cal Sans display, Inter body, Manrope landing; OFL licences are in `public/fonts` (`app/workspace.css`, `docs/workspace-redesign.md`).
  - Semantic OKLCH tokens cover task states, fact status and graph node labels.
- **Design docs contradict the code:**
  - `DESIGN.md` and `PRODUCT.md` describe a dark graphite "Chrome" theme with copper accents and Inter.
  - `.21st/DESIGN.md` describes graphite and violet.
  - The code is light cobalt.
- **Deployment:**
  - **Local:** `docker-compose.yml` runs falkordb, an api build and a web build.
  - **API image** (`apps/api/Dockerfile`): python:3.12-slim with uv 0.11.32 (`--frozen`).
    - The ONNX model is baked in and HF symlinks are flattened. `HF_HUB_OFFLINE=1`.
    - Runs non-root, with a healthcheck.
    - ~1.1 GB image; needs ~1 GB RAM with workers.
  - **Web image:** node:20-alpine with standalone output.
  - **Hosted:** `infra/README.md` gives Railway (API) and Vercel (web) recipes and says "Hosted deployment is not verified". There is **no live URL**.
- **Pitch and demo material:**
  - `docs/pitch/minutes-pitch.pptx`: 12 slides.
  - `docs/pitch/minutes-pitch-video.mp4`: 16.8 MB draft, with `pitch-script.md`, `video-storyboard.md` and `video-shots.json`.
  - `docs/demo-script.md`.
  - A LinkedIn post draft (.docx).
  - `docs/video/render_walkthrough.py`: Pillow + FFmpeg render of the landing video.
  - `docs/bench/`: 15 JSON runs.
  - `docs/audit/`.
  - About 17 MB of media in total under `docs/pitch`.

### 7.11 Licence and Windows
- **Licence:** there is no LICENSE file. The only statement is the last line of `README.md` (`:211-213`), "MIT."
- **Windows:**
  - The Python code is pure, with Windows wheels; uvloop is excluded on win32 (`uv.lock:1318`).
  - ONNX runs on CPU only, so the GPU goes unused.
  - FalkorDB needs Docker Desktop, whose daemon is currently stopped.
  - README commands use bash (`export`, `cp`). Use Git Bash or translate them for PowerShell.
  - HF symlink flattening is done only in the Dockerfile.
  - Built on our exact machine: the README bench machine is an "i9-13900HX, 48 GB RAM, Docker Desktop on WSL2". The docs mention `HF_HOME=D:\hf-cache`, and `fix-contract.md` says "Keep all storage on drive D".

### 7.12 What plugs into a live diarized-utterance stream
1. **Writes per utterance** go through `hiveboard.memory`:
   - `remember_meeting` and `remember_person` once.
   - Then `remember_episodes(db, tenant, meeting_id, [{seq, speaker_id, text, t_start, t_end, t_occurred, embedding}])` for each utterance or small batch.
   - It appends, keeps the `NEXT` chain linear (`cypher/16`, `1B`-`1D`), and publishes `episode.ingested`.
2. **Embedding:** `Embedder.embed(texts, "document")`.
3. **Extraction**, one of two ways:
   - Reuse `_extract_and_store` per window, keeping `context_before` and `container_items` across calls.
   - Submit `extract_episode` tasks to FactAgent (`ingest_episode_record`, window = 1).
4. **Live UI:** fact and supersession events over SSE.
5. **Q&A:** AnswerAgent and `why()` already cite speaker and timestamp.

**What does not work as-is:**
- `pipeline.run` raises `SourceChangedError` when a re-posted export contains new episodes, so a growing export cannot be re-posted.
- The chunker needs end-of-meeting knowledge.
- Overlapping diarized segments must be clipped first.
- Diarizer labels (`SPEAKER_00`) need a mapping to a stable `Person.id`.

### 7.13 Bugs and gaps
- **FDB-B1: No STT, diarization or audio.** `*.wav` is gitignored, there is no audio upload route, and "Play at" only highlights the transcript.
- **FDB-B2: The validator rejects raw diarizer output.** Overlapping segments, out-of-order segments, unknown speaker ids and any extra field (confidence, words) all fail. A normaliser is needed.
- **FDB-B3: Ingest is batch-only.**
  - The job store is in memory.
  - Re-posting raises `SourceChangedError`.
  - `ingest_episode_record` only re-extracts episodes that are already stored.
  - This conflicts with our "write LIVE" goal.
- **FDB-B4: Weak live extraction and a scripted demo.**
  - Strict P 0.42 / R 0.59 / F1 0.49.
  - The demo database is seeded by gold replay, so the live extraction path is not exercised.
  - `vocab.py` is hard-coded for orbit/nimbus/quill and leaked from the gold labels.
- **FDB-B5: Auth is partial.**
  - There is no per-person identity.
  - The tenant key appears as `?api_key=` on SSE and audio URLs.
  - `NEXT_PUBLIC_API_KEY` would be baked into the bundle.
  - Admin and demo routes are open unless `ADMIN_TOKEN` is set.
- **FDB-B6: Full scans per row.** Unlabelled `MATCH (m {id:$container_id})` in `cypher/10_ingest_episode`, `16_last_episode` and `1B_episode_window` cannot use an index. Likely per-utterance latency as the graph grows **[unverified, not profiled]**.
- **FDB-B7: FalkorDB 6.0.1 bug.** `MATCH…WHERE` followed directly by `MATCH` drops the filter (`docs/falkordb-issues.md`).
- **FDB-B8: Duplicated helpers.**
  - Two different fake embedders: `hiveboard/embed.py` and `ingest/embed.py`.
  - Two `fact_key`/`normalize_object` implementations: `hiveboard/memory.py` and `ingest/models.py`.
  - About 40 inline queries in `apps/api/hive_adapter.py` contradict the claim that `cypher/` is the single source of truth.
- **FDB-B9: `Turn` and `ChatSession` have no indexes.**
- **FDB-B10: Dead code.**
  - `components/pages/TenantsPage.tsx`, which holds the only ACL-probe UI.
  - `LandingSections.tsx`, `ProductTour.tsx`, `MemoryStory.tsx` and `archive/*`.
  - The unreferenced `app/experience.css` (2,671 lines) and the unused `next-themes` dependency.
  - `apps/web/README.md` still calls `/` the tenant switcher.
- **FDB-B11: Global CSS weight.** `app/layout.tsx` imports 7 landing stylesheets on every route.
- **FDB-B12: Hardcoded demo values.** `BoardPage.tsx:41,322` hardcodes "PayU integration" and `spk_priya`; `ChatPage.tsx:27` hardcodes tenant-specific suggestions.
- **FDB-B13: Doc drift.**
  - `docs/pitch/README.md` says "Nothing has been recorded yet", but the video is committed.
  - The design docs contradict the implemented theme (§7.10).
- **FDB-B14: Deployment and repo size.** Hosted deployment was never verified (`infra/README.md`). The API image is ~1.1 GB, and the repo carries ~17 MB of pitch media.

---

## 8. Port map

All of these predate the window, so every port is a **disclosed rewrite during 13-18 Oct with attribution** (§3). Reuse value comes from the input audits; where the two FalkorDB passes disagreed, both values are shown.

**How the pieces connect (from the audits):**
1. The desktop diarizer emits `{track, start, end, text}` sentences.
2. Its `_upsert`, `_relabel` and `_finalize` hooks write each utterance to FalkorDB through `hiveboard.memory.remember_episodes`, after:
   - normalising to the recorder contract (clip overlaps, map `SPEAKER_00`/track ids to a stable `Person.id`);
   - modelling the graph as Track→Person, so relabels are a single edge update.
3. The FalkorDB answer agent and Gemma client then answer with speaker + timestamp citations, and the provenance UI renders them.

The Android repo contributes the portable algorithms (Viterbi turns, token attribution) and the lighter sherpa-onnx model path.

| Component | Source paths | Reuse value | Notes |
|---|---|---|---|
| **A. Audio, diarization and attribution** | | | |
| Streaming diarization + per-speaker multi-talker ASR (`LiveSession`) | `who-said-what-desktop/wsw/engine/live.py` | High | Feeds 16 kHz PCM into NeMo `SpeakerTaggedASR`. Each 1.12 s hop runs a Sortformer step (activity (T,8) at 80 ms) and batched per-track ASR. Handles bf16/fp32 and both the multitalker and multilingual engines. Needs an NVIDIA GPU. |
| Meeting session orchestrator | `who-said-what-desktop/wsw/engine/pipeline.py` | High | Queue worker: segment upsert by (track, start), overlap flag, identify every 4 s, retroactive relabel, final pass, events. `from_file()` replays WAVs through the same path. **`_upsert` / `_relabel` / `_finalize` are the FalkorDB write hooks.** Fix DSK-B1, B5 and B6. |
| Speaker naming / scoring | `who-said-what-desktop/wsw/engine/identify.py`, `embedder.py`; Kotlin original: `who-said-what/mr/engine/SpeakerMatcher.kt` | High | Mean-centred top-3 cosine with cohort ("hub") normalisation and bounded adaptation (max 12). Desktop adds solo-frame windows, Hungarian track→person assignment and uncertain / "Speaker N" handling. Kotlin adds pairwise separation and cluster→participant assignment. Prefer the Python version. |
| Windowed Viterbi turn segmentation (`TurnLabeler`) | `who-said-what/mr/engine/TurnLabeler.kt` | High | 2 s / 0.5 s windows. The switch penalty relaxes at silences ≥1 s (×0.15) and change points (cos<0.65, ×0.3); it also does min-turn absorption and snap-to-pause. Pure math, so it ports to numpy. Only needed on the sherpa-onnx path (Sortformer already segments). |
| Token-to-turn attribution + three-pass relabel | `who-said-what/mr/engine/Pipeline.kt` (`assignTokens`, slicing, pass 1/2/3) | High | The "who said what" core on the sherpa path. Tokens are attributed by timestamp and word pieces stay with their word. Provisional → 20 s block → final relabel is the model for live-write-then-revise. Fix AND-B1, B4 and B5. |
| Windows NeMo compatibility patch | `who-said-what-desktop/wsw/engine/compat.py` | High | Swaps `torch.compile`'d flex_attention for the eager kernel on win32, quiets NeMo logging, and adds a `cuda_summary()` that never raises. Fragile across NeMo/torch upgrades. |
| sherpa-onnx engine config + model download recipe | `who-said-what/mr/engine/SherpaEngine.kt`; `who-said-what/scripts/download-models.ps1` | High | Exact files and URLs from the sherpa-onnx v1.13.8 releases: silero_vad, Parakeet-TDT 0.6B v2 int8, TitaNet-small, ERes2Net. Also VAD, transducer and greedy params, and the √2 ensemble trick. A lighter alternative to NeMo, but the non-Android bindings on this Windows/GPU setup are **[unverified]**. |
| End-of-meeting refine pass | `who-said-what-desktop/wsw/engine/finalize.py`, `offline.py` | Medium | Offline Sortformer, live→offline Hungarian alignment on a 10 ms grid, identity re-solve, onset snapping and overlap recompute. Keeps segment ids. `offline.py` also has `transcribe_simple` (Parakeet-TDT with word timestamps) and `activity_to_segments`. Fix DSK-B4. |
| Model registry / ModelHub | `who-said-what-desktop/wsw/engine/models.py` | Medium | Model specs (HF repo, file, size, licence), resumable `hf_hub_download`, and load-once shared instances via NeMo `restore_from`. Strip the neutral-name layer. Model IDs are **[unverified]**. |
| Audio capture + file loader | `who-said-what-desktop/wsw/audio/capture.py`; `who-said-what/mr/audio/Audio.kt` | Medium (desktop) / Low (Android) | Desktop: WASAPI mic + loopback, soxr, drift alignment, `Ducker`, stall and device-loss handling, streaming WavWriter. **`read_wav_mono`** (any WAV/FLAC → 16 kHz mono) is directly useful for an upload path. Live capture is Windows-only. The Android code is a pattern only (assumes a 44-byte header, AND-B6). |
| Enrollment quality gates + recorder | `who-said-what/mr/engine/Enrollment.kt`; `who-said-what-desktop/wsw/engine/enroll.py`, `wsw/server/enroll_flow.py` | Medium | Gates on speech ≥10 s, SNR ≥15 dB, clipping ≤0.5 %, phrase match ≥60 %, self-consistency ≥0.60 and separation ≥0.15, with user-facing fix messages. Desktop adds an auto-stop state machine, best-of-three and **English + Hindi phrases**. Useful for voice-first onboarding, but adds demo friction. |
| WAV replay / demo harness | `who-said-what/mr/ui/MainActivity.kt` + `mr/service/RecordingService.kt` (`debugWav`, ~4x real time, auto-enroll N voices via spectral clustering); `who-said-what-desktop` `MeetingSession.from_file`, `wsw process FILE` | Medium | Judges need a reproducible input, so recreate this as the demo and test path. |
| Spectral clustering (fixed-k) | `who-said-what/mr/engine/SpectralClustering.kt` | Low | Wang-2018 affinity refinement, Jacobi eigen-solver and k-means++. Only the debug harness uses it; sklearn or numpy would replace it. |
| **B. Storage and graph** | | | |
| Recorder export contract + validator | `FalkorDB/packages/ingest/ingest/recorder_format.py`; `FalkorDB/docs/contracts/recorder-format.md` | High | The exact bridge from diarizer to graph. It must be **relaxed or fronted by a normaliser**: it rejects overlaps, unknown ids and extra fields (FDB-B2). |
| Episode chunker | `FalkorDB/packages/ingest/ingest/chunker.py` | High (product pass) / Medium (engine pass) | Merges same-speaker segments into 30-60 s episodes and keeps the timestamps for citation. Batch-only: it needs a streaming variant for live writes. |
| FalkorDB schema + Cypher library | `FalkorDB/cypher/00_schema.cypher`, `01`-`05`, `10`-`19`, `1A`-`1D`, `30_recall_hybrid`, `31_belief_at_time`, `32_why_path`, `50_who_knows`, `51_open_commitments`, `52_contradictions`, `54_owners`; `packages/hiveboard/hiveboard/queries.py`, `schema.py` | High | Range, full-text and HNSW 768-d indexes; unique id constraints; parameterised provenance, supersession and who-knows/owns queries. Fix the unlabelled `MATCH` (FDB-B6) and keep `WITH` barriers (FDB-B7). |
| Hive Board memory substrate | `FalkorDB/packages/hiveboard/hiveboard/memory.py`, `db.py`, `queries.py`, `records.py`, `temporal.py`, `ids.py`, `errors.py`, `schema.py` | High | `remember_meeting/person/episodes` is an **incremental append** with a linear `NEXT` chain, so it is the live-write entry point. Also provides atomic `remember_fact`/`supersede_fact`/`insert_fact_in_chain`, `belief_at`, `why()`, hybrid recall and full-text, plus a lazy connection with timeouts. |
| Supersession / storage logic | `FalkorDB/packages/ingest/ingest/pipeline.py` (`_store_item`, `_belief_in_chain`, `SINGLE_VALUED_PREDICATES`, `_EntityCache`, `mentioned_known_entities`) | High | Decides confirm vs supersede vs mid-chain insert vs multi-valued add. Content-hash fact ids; entity mention linking. |
| Ingestion orchestration + LLM extraction (v5) | `FalkorDB/packages/ingest/ingest/pipeline.py`, `extract.py`, `prompts/extract_v5.md`, `vocab.py`, `dates.py`, `models.py`; `apps/api/routes/meetings.py`; `apps/api/jobs.py` | Medium | Validate → chunk → embed → store → windowed extraction → supersession → events, with async job progress. Needs an incremental path (FDB-B3). Extraction F1 is 0.49, and `vocab.py` is tenant-specific and leaked from gold (FDB-B4). |
| `hive.score` retrieval UDF | `FalkorDB/packages/hiveboard/hiveboard/udf/score.js`, `udf/loader.py` | Medium | JS UDF ranking recency × confidence × similarity, auto-loaded. |
| Transcript stores (local fallback / schema template) | `who-said-what-desktop/wsw/store/db.py`; `who-said-what/mr/data/Db.kt` | Medium | SQLite FTS5 with triggers, embedding blobs, bm25 and auto-snapshots (desktop); Room `Meeting`/`Segment`/`Speaker` (Android). Useful as templates for node properties: confidence, uncertain, overlap, provisional, words. |
| Multi-tenancy with FalkorDB ACLs | `FalkorDB/packages/hiveboard/hiveboard/tenancy.py`, `apps/api/auth.py` | Low | Graph-per-tenant, rw/ro ACL users, `GRAPH.COPY` snapshots, admin token, hashed keys. |
| **C. Models, LLM and retrieval** | | | |
| EmbeddingGemma ONNX embedder + embedding-space guard | `FalkorDB/packages/ingest/ingest/embed.py`; `pipeline.check_embedding_model`; `apps/api/ingest_service.ensure_embedding_space`; `apps/api/Dockerfile` | High | CPU ONNX, 768-d, query/document prefixes, offline-first HF cache, Gemini and fake fallbacks. Records each tenant's model and refuses mixing. Model ID **[unverified]**. |
| Gemma 4 multi-backend LLM client | `FalkorDB/apps/api/llm/gemma.py`, `__init__.py`, `fake.py`; `packages/ingest/ingest/llm_client.py`, `llm_protocol.py`; `docs/contracts/llm.md` | High | Ollama (`gemma4:e4b`), Gemini API (`gemma-4-26b-a4b-it`) and OpenRouter. JSON mode with one validated retry, native or JSON-protocol tool loop, token bucket, key rotation, `thinking_level`. Model IDs **[unverified]**. |
| Grounded answer agent + router | `FalkorDB/apps/api/agents/answer_agent.py`, `evidence.py`, `router.py`, `prompts/answer.md`; `apps/api/routes/chat.py` | High | Intent router → templates, graph, vector + full-text RRF, or guarded read-only Cypher. Citation validation with grounded fallback; chat sessions as Turn nodes (add indexes, FDB-B9). |
| Meeting Q&A with `[mm:ss Name]` citations + map-reduce notes | `who-said-what-desktop/wsw/nlp/llm.py` (`retrieve`, `ask`, `summarise`, prompts, `_CITE`); `wsw/server/llm_routes.py` | High | BM25 ±1-line retrieval, forced `[mm:ss Name]` citations, per-speaker decisions and action items. Has a **GPU eviction / idle-unload policy for sharing 8 GB VRAM** with ASR. Rebase onto Gemma 4. Single-meeting and lexical only. |
| Lexical text tools (BM25, TF-IDF, name extraction) | `who-said-what/mr/nlp/TextTools.kt`; `who-said-what-desktop/wsw/nlp/text.py` | Medium | Dependency-free BM25 (k1 1.5, b 0.75), TF-IDF summaries and word_overlap. Desktop adds **English/Hindi/Hinglish self-intro name extraction**, which needs the `strong_only` false-positive fix (DSK-B1). Good lexical baseline next to graph retrieval. |
| MCP server over the brain | `FalkorDB/packages/hiveboard/hiveboard/mcp_server.py` | Medium | recall, why, belief_at, list_tasks, claim, handoff and open_commitments over stdio. Inject a real embedder. |
| **D. API, events and UI** | | | |
| Event stream (Redis pub/sub → SSE) | `FalkorDB/packages/hiveboard/hiveboard/events.py`, `EVENTS.md`; `apps/api/bus.py`; `apps/api/routes/board.py`; `apps/web/lib/useEvents.ts`, `lib/bus.ts` | High | One long-lived subscription fanned out as SSE with replay and keep-alive, plus a React hook. The basis for showing live graph writes. Move the API key out of the URL (FDB-B5). |
| Transcript viewer with speaker colours and play-at | `FalkorDB/apps/web/components/pages/SourcesPage.tsx` (MeetingDetail, MeetingsTab, ImportPanel) | High | Speaker chips, mm:ss episodes, Play-at seek or highlight, playback follow, `?meeting=&t=` deep links. Needs real audio. |
| Provenance UI | `FalkorDB/apps/web/components/SourceLink.tsx`, `WhyPanel.tsx`, `Overlays.tsx`, `overlay-context.tsx`, `GraphView.tsx`, `graph/ForceGraphInner.tsx`, `CypherBlock.tsx`, `StateBadge.tsx` | High | "<Speaker> said in <Meeting> at mm:ss [Play at]" rows, a Why sheet with supersession history, a force-graph view and copyable Cypher. Directly serves "citing speaker + timestamp". |
| Chat page with "Why this answer" | `FalkorDB/apps/web/components/pages/ChatPage.tsx`; `apps/web/lib/api.ts`, `lib/types.ts` | High | Session-persisted chat with an evidence panel. A natural host for the **voice-first input layer, which no repo has**. Remove the hardcoded suggestions (FDB-B12). |
| React live transcript + citation UI | `who-said-what-desktop/ui/src/pages/Live.tsx`, `ui/src/components.tsx`, `ui/src/components/AskPanel.tsx`, `ui/src/api.ts`, `ui/src/pages/MeetingView.tsx` | Medium | Live per-speaker meters and an auto-scrolling transcript with provisional/overlap/uncertain badges. AskPanel turns `[mm:ss]` citations into clickable audio seeks. Typed client with WebSocket reconnect. |
| Local API + WebSocket bus + exports | `who-said-what-desktop/wsw/server/api.py`, `wsw/app.py` | Medium | FastAPI REST + thread-safe `/ws` broadcast, token auth, meeting CRUD, rename/learn-voice, MD/SRT/JSON export. Fix DSK-B2, B3 and B10. |
| Workspace shell, command menu, API status | `FalkorDB/apps/web/components/AppShell.tsx`, `CommandMenu.tsx`, `Page.tsx`, `components/ui/*`; `app/globals.css`, `app/workspace.css` | Medium | Navigation, Ctrl+K navigate-or-ask, and a health indicator that labels demo/fake/mock modes (good for honest disclosure). |
| Timeline belief-at-T | `FalkorDB/apps/web/components/pages/TimelinePage.tsx`; `cypher/31_belief_at_time.cypher`, `70_record_snapshot.cypher` | Medium | Slider comparing what was believed at time T with now, plus snapshots. Fits Track 2 O2 (memory). |
| Decision impact ("ripple effect") | `FalkorDB/cypher/80_import_work_manifest`..`83_impact_drafts`; `packages/hiveboard/hiveboard/impact.py`; `apps/web/components/pages/CompanyOverview.tsx`; `components/impact/WorkItemDialog.tsx`; `components/memory/DecisionCanvas.tsx` | Medium | WorkItem `DEPENDS_ON` Fact; a superseded fact flags work for review and saves a local, unsent follow-up draft. |
| Multi-agent board (CAS claims, handoffs, leases, skills) | `FalkorDB/packages/hiveboard/hiveboard/board.py`, `skills.py`, `analytics.py`; `cypher/20`-`2B`, `40`-`46`, `60`-`61`; `apps/api/workers.py`; `apps/api/agents/*_agent.py`; `components/pages/BoardPage.tsx`, `SkillsPage.tsx` | Low | Relevant only if pitching Gemma multi-agent (G1). Hardcoded demo values (FDB-B12). |
| Landing visuals | `FalkorDB/apps/web/components/landing/*`, `components/memory/MemoryScene.tsx`, `public/media/minutes-walkthrough.*`, `docs/video/render_walkthrough.py` | Low | Branded "Minutes"; only the patterns transfer. Heavy global CSS (FDB-B11). |
| Glass Compose UI kit | `who-said-what/mr/ui/Glass.kt` | Low | Compose-only. Carry over only the ideas: per-speaker colour hashing and a dimmed "Name ?" for uncertain lines. |
| **E. Evaluation, tests and ops** | | | |
| Diarization eval harness | `who-said-what-desktop/scripts/synth_meeting.py`, `eval_finalize.py`, `enroll_check.py`, `audio_check.py` | Medium | Builds a synthetic overlapping 4-speaker LibriSpeech meeting (dev-clean, not present) and scores frame attribution and word recall/precision before and after refine. `audio_check.py` is stale (DSK-B8). |
| Extraction eval + gold format + synthetic corpus | `FalkorDB/packages/ingest/ingest/eval.py`, `gold.py`; `data/gold/*.json`; `data/meetings/*.json`; `data/generator/*.py`; `data/seed.py`; `docs/eval/extraction.md` | Medium | Strict and partial P/R/F1 and recorder-format fixtures. Needs new gold labels for real diarized audio. |
| Test suites | `who-said-what-desktop/tests/*.py` (141 tests, no GPU); FalkorDB `packages/*/tests`, `apps/api/tests` (367 tests) | Medium | Patterns for TestClient + fakes. FalkorDB hiveboard tests need a live FalkorDB. |
| Container build with baked embedding model | `FalkorDB/apps/api/Dockerfile`, `apps/web/Dockerfile`, `docker-compose.yml`, `infra/README.md` | Medium | Reproducible ~1.1 GB API image, standalone Next image, FalkorDB 6.0.1 compose, and Railway/Vercel recipes (hosted deployment unverified). |
| Benchmark harness | `FalkorDB/packages/hiveboard/bench/harness.py`, `docs/bench/*.json` | Low | Multi-process mixed read/write + CAS race. Recorded 6.0.1 vs 4.22.0 runs on this same laptop. |
| RESEARCH.md library survey | `who-said-what/RESEARCH.md` | Medium | 2026-09-25 survey of on-device ASR, diarization and speaker-ID options (sherpa-onnx, whisper.cpp, Moonshine, pyannote, FluidAudio, Picovoice and others) with source URLs. Claims are third-party and **[unverified]**. |
| Desktop packaging / first-run setup | `who-said-what-desktop/wsw/launcher.py`, `wsw/server/setup_flow.py`, `installer/WhoSaidWhat.iss`, `scripts/*.ps1`, `ui/src/components/SetupOverlay.tsx` | Low | Not relevant to a web deployment. Its per-GPU CUDA wheel selection logic is the only useful reference. Unchecked uv download (DSK-B12). |
| Model-name obfuscation | `who-said-what-desktop/wsw/neutral.py` | Low (drop) | Hides model and framework names, which counts against an open-source, verifiable entry (§3.3). |

---

## 9. Bugs, gaps and risks

Repo-specific code bugs are listed with IDs in §5.7 (AND-), §6.7 (DSK-) and §7.13 (FDB-). This section is the cross-cutting risk register, grouped by how directly each risk threatens the submission.

### 9.1 Submission blockers
1. **R1: Prior-art timing (all repos).**
   - Every commit predates the 13-18 Oct window: 2026-09-25 → 2026-10-08, 2026-09-26 → 2026-09-27, and 2026-10-03 → 2026-10-12.
   - Verbatim reuse breaches the window rule (`handbook :186-188`) and the original-work rule (`:259`).
   - Mitigation: disclosed rewrites in a new repo (§3.3).
2. **R2: Concurrent Graph Hacks submission.** `FalkorDB` targets Graph Hacks (Oct 15-18, 2026), which overlaps our event. Its own README and `docs/track3-readiness.md` ask for an organiser ruling, and whether the team still enters is **[unverified]**. Overlapping submissions could disqualify the team from either event.
3. **R3: No LICENSE in any repo.**
   - `who-said-what` is "all rights reserved".
   - `who-said-what-desktop` claims MIT only in `pyproject.toml:7`.
   - `FalkorDB` claims MIT only in a README line.
   - Owners must relicense before public reuse.
   - Model obligations: CC-BY-4.0 attribution (Parakeet-TDT, TitaNet-small), and a redistribution check for NVIDIA Open Model and OpenMDW-1.1 **[unverified]**.
4. **R4: Live deployment URL.** No repo has a verified hosted deployment.
   - The Android app cannot run on the web.
   - The desktop pipeline needs a CUDA GPU (no CPU mode) and Windows-only capture.
   - The FalkorDB API is a ~1.1 GB container that needs ~1 GB RAM and a private FalkorDB host. Vercel can host only the Next.js web app.
   - Hosting ASR plus embedding models on free Vercel/Netlify/Render tiers is likely infeasible **[unverified tier limits]**.
   - The inputs suggest a split architecture: hosted graph/Q&A UI plus a local or recorded diarization feed.
5. **R5: Accurate contribution description.**
   - `wsw/neutral.py` hides model names (DSK-B11).
   - The FalkorDB demo relied on gold replay and scripted answers (FDB-B4).
   - FalkorDB has non-lead authors (PDGamerSG, "Admin").
   - All of this must be disclosed and credited (`handbook :259`).

### 9.2 Architecture gaps for "write LIVE into FalkorDB"
6. **R6: No live per-utterance ingest path exists.** FalkorDB ingest is batch per meeting with an in-memory job store. Re-posting raises `SourceChangedError`, and the chunker needs the whole meeting (FDB-B3). A new streaming writer must call `hiveboard.memory.remember_episodes` directly (§7.12).
7. **R7: Labels are revised after the fact in both diarizers.**
   - Android deletes and re-inserts rows (AND-B4).
   - Desktop relabels and refines, and its `(track,start)` upsert key can drift (DSK-B6).
   - A schema that stores the speaker on each utterance needs mass rewrites. Use `(:Utterance)-[:ON_TRACK]->(:Track)-[:RESOLVED_AS]->(:Person)` or an upsert/provisional design.
8. **R8: Contract mismatch.** `recorder_format.py` rejects overlapping and out-of-order segments, unknown speaker ids and extra fields. Raw diarizer output (overlaps, `SPEAKER_00`, confidence/words) fails without a normaliser (FDB-B2).
9. **R9: No voice-first Q&A anywhere.**
   - Searched all three repos for speechSynthesis, text-to-speech, TTS, piper, SpeechRecognition, getUserMedia and MediaRecorder.
   - The only hits are Android mic capture (`mr/audio/Audio.kt`) and a Windows-TTS test clip generator (`who-said-what-desktop/scripts/enroll_check.py`).
   - FalkorDB has no voice input or TTS. A G2 voice-first layer would be new work.
10. **R10: Retrieval scope.** Desktop `ask()` is lexical BM25 within one meeting. Cross-meeting, semantic "company brain" retrieval exists only in FalkorDB, whose extraction is weak (R16).

### 9.3 Hardware, environment and runtime
11. **R11: VRAM budget on the 8 GB RTX 4060.** Live diarization + ASR measured 3.2-3.4 GB, and the repo's 4B Q4 LLM added ~3.4 GB. Gemma 4 sizes and quantisations are **[unverified]**, so live ASR and Gemma 4 may not fit together; the eviction policy in `llm_routes.py` hints at this.
12. **R12: Heavy, fragile environment.**
   - torch 2.14.0 from the CUDA index, plus NeMo from a pinned git archive.
   - ~6 GB install; first NeMo import ~4 min.
   - Required models ~2.8 GB, plus 2.4 GB for multilingual and 2.5 GB for offline ASR.
   - Nothing is pulled yet: no Ollama models, the Docker daemon is stopped, and the FalkorDB image is not pulled.
   - Download and install time must be planned inside the window.
13. **R13: NeMo is not officially supported on Windows.** It works only through the `compat.py` flex_attention monkey-patch, which could break with any NeMo or torch change.
14. **R14: Model footprint on the sherpa path.** Parakeet-TDT 0.6B v2 int8 is ~640 MB on disk and ~1 GB RAM, plus two embedding models. Whether sherpa-onnx 1.13.8 Python/CUDA works on this machine is **[unverified]**.
15. **R15: Latency and quota.**
   - Gemma answers took 6-17 s.
   - The Gemini free tier caps input at 16k tokens per minute (repo claim), and the eval paced to ~4 calls/min.
   - Ollama `gemma4:e4b` latency on the RTX 4060 is unmeasured.
   - FDB-B6 full scans may slow per-utterance writes **[unverified]**.

### 9.4 Quality and trustworthiness
16. **R16: Extraction quality.** Strict P 0.42 / R 0.59 / F1 0.49 (partial P 0.49 / R 0.69). The demo DB was gold-seeded, and the vocabulary is tenant-specific and leaked from gold (FDB-B4).
17. **R17: Attribution errors.**
   - Android: no "unknown" rejection (AND-B2); Quick start forces saved names (AND-B1); utterances under 0.8 s inherit labels; overlap is unsupported (AND-B5).
   - Desktop: self-intro aliasing creates bogus Person nodes such as "Going" and "Sure" (DSK-B1).
18. **R18: Uncalibrated confidence.** Android `confidence` is a cohort-normalised score, constant 1.0 in open set (AND-B3). Do not present it as verifiable confidence, which matters for Track 2 O3.
19. **R19: Language coverage.**
   - Parakeet-TDT 0.6B v2 is English-only, and the suggested v3 covers European languages, not Indic ones.
   - The desktop multilingual engine masks rather than separates overlaps, and Hindi/Hinglish accuracy is not evaluated.
   - Partial help: Hindi enrollment phrases and Hinglish name extraction exist in the desktop repo.
20. **R20: Tuning and benchmarks.** Thresholds were tuned on narrow data: sherpa-onnx's two-speaker file (Android) and synthetic LibriSpeech (desktop). All performance figures (182 ms/step, 97.9 % attribution, RTF/WER in `RESEARCH.md`, FalkorDB latencies) are author-reported and **were not re-measured**.
21. **R21: Unverified model IDs.** `gemma4:e4b`, `gemma4:12b`, `gemma-4-26b-a4b-it`, `google/gemma-4-26b-a4b-it`, `onnx-community/embeddinggemma-300m-ONNX`, `gemini-embedding-001`, and the NeMo IDs in `wsw/engine/models.py` are all repo-declared and **[unverified]** against Ollama, Google, OpenRouter and Hugging Face.

### 9.5 Security and privacy
22. **R22:** Desktop SPA catch-all plausibly allows local arbitrary file reads on Windows (DSK-B3). The settings endpoint is unvalidated (DSK-B10), and uv is downloaded without a checksum (DSK-B12).
23. **R23:** FalkorDB auth gaps (FDB-B5): no per-person identity, the API key in SSE and audio URLs, `NEXT_PUBLIC_API_KEY` in the bundle, and admin routes open without `ADMIN_TOKEN`.
24. **R24:** Voice embeddings are biometric data. Storing them in a hosted graph requires consent and privacy handling.

### 9.6 Engineering hygiene
25. **R25: FalkorDB 6.0.1 query-shape bug** (FDB-B7). New Cypher must use `WITH` barriers or it returns quietly wrong results.
26. **R26: Doc/code drift in every repo.**
   - DSK-B9, plus the stale scripts DSK-B7 and DSK-B8.
   - FDB-B13 and the contradicting design docs.
   - FDB-B8 duplicated helpers and inline Cypher.
   - The docs cannot be taken at face value.
27. **R27: Branding and dead code.** Everything is branded "Minutes" or "Who Said What". FalkorDB dead code would carry over if folders were copied wholesale (FDB-B10, FDB-B11): `TenantsPage.tsx`, the 2,671-line `experience.css`, and 7 global landing stylesheets.
28. **R28: Live-demo friction.**
   - Enrollment needs 10-24 s per person and must pass the quality gates.
   - First NeMo import takes ~4 min.
   - A replay-WAV demo path is needed (port map, harness row).
29. **R29: Untested paths.** Android has no tests (AND-B10). FalkorDB's live extraction and the hosted deploy have never been exercised. Android uses destructive migrations (AND-B8).

---

## 10. Audit notes, corrections and open items

**Re-verified in this merge** (read-only commands only; nothing written under `reference/`):
- **Commit counts, ranges and authors.** `git log --date=iso` on all three clones matches the inputs: 3 / 40 / 44 commits, dates as in §3.2, and author date = committer date.
- **Handbook rule lines.** `docs/handbook/00-handbook.md:32, 186-188, 192-194, 256-259`.
- **"Falcon".** `grep -ril falcon reference docs` returns only `who-said-what/RESEARCH.md:95,250` (Picovoice).
- **"dataison".** `dataison|datason|dataisan|data-ison` returns 0 hits.
- **`cypher/` file count.** The FalkorDB product pass said "about 60" and the engine pass said 59; `ls cypher/*.cypher` = **59**.
- **FalkorDB README.** `README.md:5,7` (Graph Hacks target, pre-event status) and `:211-213` (licence "MIT.").
- **Voice/TTS search** for R9.
- **Web component paths** in §4.3 B, checked against `git ls-files`.

**Corrections to the input audits:**
- **Android "Retrieve" search.** The `who-said-what` pass reported **0** hits for "Retriev". A re-run of `git grep -i retriev` finds **1 generic prose hit**, `RESEARCH.md:17` ("BM25 keyword retrieval"). The conclusion (no branding) is unchanged.
- **Graph Hacks timing.** "Earlier FalkorDB hackathon" is accurate only for the commits (Oct 3-12). The event itself, Graph Hacks Oct 15-18, 2026, is concurrent with ours.

**Not done / open:**
- **No build, install or run** for any repo. Performance numbers, test pass counts and hosted behaviour are as reported by the repos.
- **Desktop SPA path-traversal** (DSK-B3) was not exercised against a running server.
- **Media not inspected.** FalkorDB images and videos were not OCR'd or transcribed for the "Retrieve" check.
- **Decisions the team needs to make:**
  - (a) Are we also entering Graph Hacks with the FalkorDB repo?
  - (b) Will the owners add LICENSE files before we port?
  - (c) Is "Retrieve" intended as our new product name?
  - (d) Do we take the desktop NeMo path or the sherpa-onnx path for diarization, given VRAM and deployment constraints?
