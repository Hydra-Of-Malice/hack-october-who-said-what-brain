# Deploying the live URL

The handbook requires "a working live deployment URL where judges can test the solution". Three ways to get one are listed below, from most to least robust. Every option serves the same single app: FastAPI plus the built UI on one port.

## What the hosted app needs

| Piece | Hosted choice | Notes |
|---|---|---|
| App container | `Dockerfile` (about 1.5 GB RAM at runtime) | Bakes in the VAD, speaker-embedding and Whisper models |
| FalkorDB | Render private service with a disk, Railway's FalkorDB template, or FalkorDB Cloud | Set `WSW_FALKORDB_URL`. Do **not** rely on FalkorDB Cloud Free for judging: it has no persistence and stops after a day idle (TECH_NOTES §2.6). The app reseeds the demo meetings on boot when the graph is empty, which softens this |
| Gemma 4 | `GEMINI_API_KEY` → `gemma-4-31b-it` on the Gemini API (free tier) | Hosted Gemma 4 has no audio input, so on the hosted URL spoken questions go through Whisper and then Gemma (the UI says which path ran). The laptop demo uses Gemma 4 E4B native audio |

## Option A: Render Blueprint (recommended)

1. Go to render.com → New → Blueprint, and select `Hydra-Of-Malice/hack-october-who-said-what-brain`.
2. When prompted, set `GEMINI_API_KEY` (from https://aistudio.google.com/apikey).
3. Deploy. `render.yaml` creates `wsw-falkordb` (a private service with a 1 GB disk) and `wsw-brain` (a Docker web service on the Standard plan).
4. Open `https://wsw-brain.onrender.com/api/health`. Expect `falkordb.ok: true` and `llm.provider: "gemini"`. On first boot the demo meetings are ingested through the real listener, which takes a few minutes on a shared CPU.

## Option B: any Docker host (DigitalOcean App Platform or Droplet, Fly, Railway)

```bash
docker build -t wsw-brain .
docker run -p 8000:8000 -e GEMINI_API_KEY=... -e WSW_FALKORDB_URL=falkor://default:PASS@HOST:PORT wsw-brain
```

Or use `docker compose up --build`, which also starts FalkorDB.

## Option C: share the laptop (zero accounts, venue fallback)

This option runs everything on the presenter laptop, including Gemma 4 E4B native audio through Ollama, and exposes it over HTTPS with a no-signup tunnel. HTTPS matters because browsers allow the microphone only on HTTPS or localhost. The URL works while the laptop is on.

```powershell
# 1. FalkorDB (WSL)
wsl -d Ubuntu-24.04 -- bash "/mnt/d/HACKATHONS/Hack October/scripts/falkordb-wsl.sh" start
# 2. App (serves the built UI on :8000)
cd apps/web; npm run build; cd ../api; uv run python -m wsw_brain --port 8000
# 3. Public HTTPS URL
powershell -File scripts/share.ps1
```

`scripts/share.ps1` uses `cloudflared` if it is installed (`winget install Cloudflare.cloudflared`), and otherwise falls back to `ssh -R` through localhost.run. Neither needs an account. The printed `https://…` URL is the one to submit if no cloud deploy is available.

## Checklist before submitting the URL

- [ ] `/api/health` reports `ok: true`, FalkorDB ok, and the expected Gemma model.
- [ ] The meetings list shows the two demo meetings with named speakers.
- [ ] Asking "What did Arjun commit to?" (typed or spoken) returns an answer with citations that play.
- [ ] The URL is opened from a phone on mobile data (not the venue Wi-Fi) to confirm it is public.
