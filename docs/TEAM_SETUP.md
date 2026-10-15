# Team setup: get it running on your machine

This takes about 20 minutes on a fresh machine, mostly downloads. The rest of the README tells you what the product is. This page tells you how to work on it.

## 1. Get the code

```bash
git clone https://github.com/Hydra-Of-Malice/hack-october-who-said-what-brain.git
cd hack-october-who-said-what-brain
cp .env.example .env        # every value has a working local default
```

## 2. Install the tools

| Tool | Why | Install |
|---|---|---|
| Python 3.12 + [uv](https://docs.astral.sh/uv/) | API, listener, Gemma adapter | `pip install uv` or the uv installer; uv fetches Python 3.12 itself |
| Node 20+ | web UI | nodejs.org |
| [Ollama](https://ollama.com) + Gemma 4 E4B | local Gemma 4, including **native audio** for spoken questions (about 7 GB) | `ollama pull gemma4:e4b` |
| FalkorDB | the graph | pick one option from the list below |

**FalkorDB options:**
- **Docker:** `docker compose up falkordb` (port 6379, and the FalkorDB Browser on 3000).
- **Windows without Docker:** `wsl -d Ubuntu-24.04 -- bash scripts/falkordb-wsl.sh start`. It runs FalkorDB from the `falkordblite` wheel inside WSL.
- **FalkorDB Cloud:** put its URL in `.env` as `WSW_FALKORDB_URL=falkor://default:<password>@<host>:<port>`.

**No GPU, or no Ollama?** Set `GEMINI_API_KEY` in `.env` (from https://aistudio.google.com/apikey). Answers then use hosted `gemma-4-31b-it`, and spoken questions go through Whisper first.

## 3. Install and run

```bash
uv run --project apps/api python scripts/fetch_models.py   # VAD + speaker-embedding ONNX into ./models (80 MB)
cd apps/web && npm ci && npm run build && cd ../..
uv run --project apps/api python -m wsw_brain --port 8000
```

Open http://localhost:8000. On first start the two demo meetings in `demo/audio/` are ingested through the real listener, which takes about 3–4 minutes with Ollama. `/api/health` shows what is connected.

The first run also downloads faster-whisper-small and EmbeddingGemma-300m into your Hugging Face cache:
- If the EmbeddingGemma download asks for a login, run `hf auth login` once.
- Or set `WSW_EMBEDDER=hash` to use the fallback embedder.

## 4. Day-to-day development

| Task | Command |
|---|---|
| API with auto-reload | `uv run --project apps/api python -m wsw_brain --port 8000 --reload` |
| UI with hot reload | `cd apps/web && npm run dev`, then open http://localhost:5173 (it proxies `/api` and `/ws` to :8000) |
| Replay a meeting as if it were live | `uv run --project apps/api python scripts/stream_wav.py demo/audio/02-launch-sync.ogg` |
| Backend tests | `cd apps/api && uv run pytest -q` (add `-m "not live"` to skip tests that need Ollama) |
| Lint | `cd apps/api && uv run ruff check wsw_brain`, and `cd apps/web && npx tsc -b && npx oxlint src` |
| Reset the demo graph | stop the server, then `uv run --project apps/api python -c "from falkordb import FalkorDB; FalkorDB(host='127.0.0.1', port=6379).select_graph('wsw_demo').delete()"`, then start it again (it reseeds) |

## 5. Where things live

| Path | What |
|---|---|
| `brand.json` | product name (the only place it is defined) |
| `apps/api/wsw_brain/listen/` | who-said-what engine: VAD, speaker embeddings and clustering, Whisper |
| `apps/api/wsw_brain/graph/` | every FalkorDB query (`cypher.py`) and the store |
| `apps/api/wsw_brain/brain/` | Gemma 4 adapter, voice agent and tools, extraction, retrieval, EmbeddingGemma |
| `apps/api/wsw_brain/api/` | REST, SSE, audio WebSocket, uploads, demo seeding, access rules |
| `apps/api/wsw_brain/pipeline.py` | live ingest: utterances → graph batches → extraction |
| `apps/web/src/` | React UI; design tokens are in `styles/tokens.css` (see [DESIGN.md](DESIGN.md)) |
| `demo/` | demo meeting scripts, audio, ground truth, diarization scorer |
| `docs/` | hackathon brief, strategy, API contract, deploy guide, submission checklist |

## 6. Working together

- **Branch and PR.** Work on a feature branch and open a PR to `main`. Keep PRs focused, well under 5,000 changed lines. This is also how the ShipWright leaderboard counts work:
  - PRs must be opened by a human, reviewed by the ShipWright app and merged.
  - Pushing fixes after its review ranks you higher.
  - Someone with admin rights must first install https://github.com/apps/shipwright-agent on this repo.
- **Rules that matter for eligibility:**
  - **Only write new code.** Never copy code from the team's earlier repos (who-said-what, who-said-what-desktop, FalkorDB "Minutes"). They predate the 13–18 Oct build window. Ideas are fine; code is not (MLH rule 9). They are disclosed in the README.
  - **Never commit `reference/`, `data/`, `models/` or `.env`.** `.gitignore` already covers them.
  - **List any AI tool you use** in the README's prior-work section.
  - **No pushes after the 3:30 PM IST deadline on 18 Oct.**
- **Run the tests before merging.** Run `uv run pytest -q` and `npx tsc -b`.

## 7. Known gotchas (Windows)

- **Use `127.0.0.1`, not `localhost`, for FalkorDB.** On Windows `localhost` tries IPv6 first and waits about 2 s on every connection.
- **The Ollama tray app may ignore `OLLAMA_MODELS`.** If `ollama list` is empty but you have models on another drive, quit the tray app and run `ollama serve` with `OLLAMA_MODELS` set.
- **If Docker Desktop fails** with "The file cannot be accessed by the system" on sockets under `%LOCALAPPDATA%\Docker\run`, use the WSL FalkorDB script.
- **The microphone needs HTTPS or localhost.** Never demo over `http://<LAN-IP>`.
