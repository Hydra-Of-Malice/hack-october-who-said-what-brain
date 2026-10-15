# Demo meetings: Kitepost

Two scripted meetings at **Kitepost**, a fictional Bengaluru B2B logistics SaaS that sells shipment tracking and delivery updates to small and mid-size shippers. Each meeting is synthesised to audio with a different open-weight voice per person.

The audio goes through the **real listener**: VAD, ASR, speaker clustering, then live graph writes. There is no pre-baked transcript, so the demo shows diarization and live FalkorDB writes really working. A ground-truth file for each meeting lets us score how well the listener did.

| File | What it is |
|---|---|
| `meetings/01-pricing-review.json` | Script: the self-serve pricing review, Tuesday 13 Oct 2026. 4 speakers, 46 turns |
| `meetings/02-launch-sync.json` | Script: the launch sync two days later. 3 speakers, 29 turns. Reverses the Starter price |
| `audio/01-pricing-review.ogg` | 6 min 20 s, Ogg Opus, 16 kHz mono, 1.74 MB |
| `audio/02-launch-sync.ogg` | 3 min 56 s, Ogg Opus, 16 kHz mono, 1.08 MB |
| `meetings/<id>.truth.json` | Ground truth for each turn: `{seq, speaker, t_start, t_end, text}` in seconds, written during synthesis |
| `make_audio.py` | Regenerates the audio and the truth files from the scripts |
| `score_diarization.py` | Scores the listener's output (`GET /api/meetings/{id}`) against a truth file |
| `tests/` | Script contract, truth/audio consistency, and scorer tests |

Each script's `expected` block lists the decisions, action items (owner, text, due), disagreements, entities, figures and rehearsal Q&A that a correct run should surface.

## The cast

The voices are deliberately far apart, which helps speaker clustering. Measured with TitaNet-small embeddings on this audio:
- each turn's cosine similarity to its own speaker centroid is at least 0.95;
- its similarity to any other speaker's centroid is at most 0.26.

The listener's threshold is 0.55, so there is a wide margin.

| Name | Role | Kokoro voice | Accent |
|---|---|---|---|
| Priya | Head of Product | `hf_alpha` (speed 1.05) | Indian English (a Hindi voice reading English) |
| Arjun | Engineering Lead | `am_michael` (1.0) | American English |
| Meera | Sales Lead | `bf_emma` (1.0) | British English |
| Rahul | Finance Manager | `bm_george` (1.05) | British English |

## Meeting 1: Self-serve pricing review (Tue 13 Oct, ~6.3 min)

**Decisions:**
- **Starter tier: 4,999 rupees a month**, with 1,500 shipments included and **4 rupees per extra shipment** (02:14, Priya).
- **Annual discount: 15 percent**, not 20, reviewed after 90 days of data (03:50, Priya sides with Rahul).
- **Self-serve launch: Monday, November 2** (05:19).
- Growth stays at 14,999 rupees for 10,000 shipments. Monsoon Apparel (three warehouses) becomes an Enterprise deal.

**Disagreement:** Meera against Rahul, twice.
- Meera wants Starter at 3,999 to match Trackwise. Rahul says that is only about 32 percent margin on a busy account, against 46 percent at 4,999.
- Meera wants a 20 percent annual discount. Rahul caps it at 15, citing the 34-account Saffron Basket cohort, where 40 percent chose annual billing at 12 percent off.

**Action items:**

| Owner | What | Due | At |
|---|---|---|---|
| Arjun | Ship the usage meter (on staging Thursday night, warning at 90 percent of the plan) | Friday | 04:13 asked, 04:20 accepted |
| Meera | Send the revised deck to Saffron Basket before their Friday renewal call | Thursday | 04:27 |
| Rahul | Set up monthly and annual Starter and Growth plans in Razorpay | Wednesday | 04:36 |
| Priya | Draft the pricing page and the FAQ, with worked overage examples | Monday | 05:01 |
| Rahul | Confirm the GST registration either way | next Tuesday | 05:39 |
| Meera | Take Monsoon Apparel through the Enterprise proposal | next week | 02:54 |

**Figures people ask about:**
- cost per tracked shipment: 1.8 rupees (00:09, Rahul);
- overage: 4 rupees;
- FAQ example: 2,000 shipments on Starter come to 6,999 rupees.

**Entities:** Kitepost, Trackwise (competitor), Saffron Basket (customer), Monsoon Apparel (prospect), Razorpay (billing vendor), WhatsApp.

## Meeting 2: Self-serve launch sync (Thu 15 Oct, ~3.9 min)

Priya, Meera and Rahul attend; Arjun is busy on the usage meter.

**Reported done:** Meera sent the revised deck to Saffron Basket (00:08). This is the action item from meeting 1.

**Why the price changes:**
- Trackwise cut its plan from 3,999 to 3,499.
- Saffron Basket and two of the five new trial accounts pushed back on 4,999.
- Rahul re-ran the numbers: with 1,000 included shipments, 3,999 gives a **55 percent** margin (01:06).

**Decision that supersedes meeting 1:** Starter moves from 4,999 to **3,999 a month with 1,000 shipments** (01:50, Priya: "This replaces what we decided on Tuesday"). Overage stays at 4 rupees. The annual discount (15 percent) and the launch date (November 2) are reaffirmed.

**New action items:**

| Owner | What | Due |
|---|---|---|
| Rahul | Update the Starter plan in Razorpay to 3,999 with the 1,000-shipment limit | Monday (02:19) |
| Priya | Message Arjun so the usage meter warns at 900 shipments | today (03:05) |
| Meera | Email Saffron Basket and the two trial accounts the new Starter price | tomorrow morning (03:13) |

**Still open from meeting 1:**
- Priya's pricing page, due Monday, now with 3,999 and a 7,999 FAQ example.
- Arjun's usage meter, due Friday.
- Rahul's GST check, due Tuesday.

## Rehearsal Q&A

Ask these by voice once both meetings have been ingested.

A correct answer mentions the listed facts, cites the speaker and timestamp, and gives the newer decision when they conflict. When checking numbers, ignore commas and the currency sign: "₹3,999", "3999" and "3,999 rupees" are the same answer.

| # | Question | A correct answer mentions |
|---|---|---|
| 1 | What did Priya commit to in the pricing review? | pricing page and FAQ, by Monday (01-pricing-review 05:01) |
| 2 | And when is that due? *(follow-up)* | Monday |
| 3 | Mark it done. *(action)* | marks Priya's pricing-page item done |
| 4 | What is the Starter price now? | 3,999 with 1,000 shipments; it was 4,999 until the launch sync |
| 5 | Why did we change the Starter price? | Trackwise cut to 3,499; Saffron Basket pushback; 55 percent margin |
| 6 | Who wanted a bigger annual discount, and what did we settle on? | Meera wanted 20; Rahul capped at 15; Priya sided with Rahul |
| 7 | Who owns the usage meter and when is it due? | Arjun, Friday; warning moves to 900 shipments |
| 8 | Did Meera send the revised deck to Saffron Basket? | yes, done, reported in the launch sync |
| 9 | What does Meera have to do by tomorrow? | email Saffron Basket and the trial accounts the new price |
| 10 | What does a tracked shipment cost us? | about 1.8 rupees (Rahul) |
| 11 | What happens with Monsoon Apparel? | Enterprise proposal, Meera, three warehouses |
| 12 | Is the November second launch still on? | yes; GST registration pending, Rahul confirms by Tuesday |

The scripts list more questions under `expected.qa`.

## Running the demo audio through the app

```bash
# 1. upload (pace=realtime for the live-room effect, fast to just ingest)
curl -F file=@demo/audio/01-pricing-review.ogg -F "title=Self-serve pricing review" -F pace=fast \
     http://localhost:8000/api/meetings/upload
# 2. name the speakers in the UI (S1 -> Priya, ...), then save the meeting JSON
curl http://localhost:8000/api/meetings/<meeting_id> -o m1.json
# 3. score it
uv run --project apps/api --no-sync python demo/score_diarization.py demo/meetings/01-pricing-review.truth.json m1.json
```

The scorer prints:
- the best track-to-person mapping (Hungarian assignment on overlapping speech time);
- time-weighted speaker accuracy;
- turn-level and utterance-level accuracy;
- DER with a 0.25 s collar, and coverage;
- accuracy using the names typed in the UI, once speakers are named;
- transcript WER and word recall (numbers are normalised, so "fifteen hundred" equals "1500").

Use `--json` for the full result. `--min-speaker-acc 0.9` makes it exit 1 below that accuracy.

**Reference point.** These numbers come from a deliberately crude throwaway pipeline, not the product listener: sherpa-onnx Silero VAD, then TitaNet-small with greedy online clustering at 0.55, then faster-whisper small int8.

| Meeting | Speaker attribution | DER | Missed speech | Transcript WER |
|---|---|---|---|---|
| 01-pricing-review | 100 % (46/46 turns) | 6.5 % | VAD edge trimming only | 4.4 % |
| 02-launch-sync | 100 % (29/29 turns) | 5.8 % | VAD edge trimming only | 4.7 % |

The real listener should match or beat this.

Known ASR spellings to expect:
- "Kite Post" for Kitepost;
- "Mira" for Meera, occasionally;
- "Safran Basket", once;
- "track-wise" for Trackwise.

## Regenerating the audio

`kokoro-onnx` is **not** a project dependency. uv layers it on top of the API environment for one run, without touching `uv.lock`:

```bash
# from the repo root; --download fetches the model files on first use
uv run --project apps/api --no-sync --with kokoro-onnx python demo/make_audio.py --download
```

- The model files go to `models/kokoro/` (git-ignored):
  - `kokoro-v1.0.onnx`, 325 MB;
  - `voices-v1.0.bin`, 28 MB.
  Both come from the official kokoro-onnx GitHub release `model-files-v1.0`.
- Synthesis takes about 2.5 min for meeting 1 and 1.5 min for meeting 2 on CPU.
- Pass meeting ids to render only some meetings, for example `demo/make_audio.py 02-launch-sync`.

Other options:
- `--voice Priya=af_heart` tries a different voice;
- `--format wav` writes uncompressed audio;
- `--room-tone-db -120` gives pure digital silence between turns.

What the script does for each turn:
1. Synthesises at 24 kHz.
2. Resamples to 16 kHz with soxr (HQ).
3. Trims leading and trailing silence, so the truth timings are tight.
4. Normalises speech loudness to -20 dBFS RMS, with a -1 dBFS peak ceiling.
5. Adds the script's `pause_after_s` of silence.

Around the turns it adds a 0.6 s lead-in, a 1.0 s tail, and seeded room tone at -62 dBFS.

Re-runs decode to identical samples, but the `.ogg` bytes still change because libsndfile picks a random Ogg stream serial number. Commit new audio only when a script has changed.

Editing a script? Keep these rules; the tests check them:
- 1 to 3 sentences per turn;
- no two consecutive turns by the same person;
- `pause_after_s` between 0.3 and 0.9 (the scripts use 0.5 to 0.9, so VAD can split speakers);
- the same voice for a person in every meeting.

Then rerun `make_audio.py` so the audio and the truth files stay in step.

Tests:

```bash
uv run --project apps/api --no-sync pytest demo/tests -q
```

## Licences and disclosure

- **Voices and model.** Kokoro-82M v1.0 by hexgrad, **Apache-2.0** (https://huggingface.co/hexgrad/Kokoro-82M). The voice pack `voices-v1.0.bin` is the Kokoro v1.0 voice set, repackaged by kokoro-onnx.
- **Runtime.** `kokoro-onnx` 0.6.1, **MIT** (https://github.com/thewh1teagle/kokoro-onnx).
- **Generation-time tools.** These pull in phonemizer and espeak-ng (via `espeakng-loader`), both **GPL-3.0**. They run only when the audio is generated. They are not in this repository, not in the app's dependencies, and not shipped.
- **Our files.** The scripts, the generated audio and the truth files are original work for this project, under the repository's Apache-2.0 licence. The audio is AI-generated speech.
- **Fiction.** Kitepost, its people, Trackwise, Saffron Basket and Monsoon Apparel are fictional. Razorpay and WhatsApp are named only as generic vendors a business like this might use; no relationship is implied.

## Windows note: sherpa-onnx and onnxruntime in one process

Windows ships its own `C:\Windows\System32\onnxruntime.dll` (1.17.1). The sherpa-onnx 1.13.8 wheel does not bundle onnxruntime, so it loads that system DLL and fails with this error when a model is created:

> The requested API version [28] is not available ... Current ORT Version is: 1.17.1

The API already handles this in `wsw_brain/listen/_sherpa.py`, which preloads onnxruntime's own DLL.

A standalone script that uses sherpa-onnx (for example, to inspect speaker embeddings of the demo audio) must do one of these before `import sherpa_onnx`. Both were verified on this machine:

```python
import os, onnxruntime
os.add_dll_directory(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))
# or: ctypes.WinDLL(<that folder>/onnxruntime.dll)
import sherpa_onnx
```
