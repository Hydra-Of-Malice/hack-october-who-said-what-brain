# Design direction: "the multitrack"

**Subject.** Meetings as recorded audio: who spoke, when, and what they committed to.

**Audience.**
- Team leads and ICs looking for "who said what" after the fact.
- Judges at a 2-minute expo demo.

**Primary job.** Make speaker-attributed speech feel tangible, checkable and askable.

**The one memorable thing** is the **speaker lane**. It borrows the vernacular of a recording studio's multitrack session: one horizontal lane per speaker, with blocks where they spoke, a playhead, and click-to-jump. Lanes appear in two places:
- **Meeting room:** full lanes.
- **Meetings index:** each meeting is a single compressed "tape strip" of coloured blocks.

Everything else stays quiet so the lanes carry the identity.

## Tokens

All tokens live in [`apps/web/src/styles/tokens.css`](../apps/web/src/styles/tokens.css). Components read **only** tokens, so a restyle is a token swap.

**Colour.** A cool studio-desk grey. This deliberately avoids the cream-and-terracotta and black-and-acid defaults.

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#E3E6E8` | desk grey behind panels |
| `--surface` | `#F7F8F9` | panels, transcript |
| `--ink` | `#13202B` | text, primary buttons (navy ink, not #111) |
| `--ink-2` | `#46535F` | secondary text (AA on surface) |
| `--line` | `#C3CAD0` | dividers, panel borders |
| `--rec` | `#C8261B` | **live / recording only**, never decorative |
| `--focus` | `#2350C8` | focus rings, links |

**Speaker palette.** Eight tape-label hues, assigned by speaker index: cobalt `#2F5BD3`, teal `#0E8577`, mustard `#B98900`, plum `#A23B84`, moss `#5E7F1E`, rust `#B4531C`, violet `#6A4FC9`, slate `#4B6175`. Each colour is used for:
- the lane block;
- the transcript's speaker bar;
- citation quote rules.

Text over speaker colour is always white, and every hue passes AA for white text.

**Type:**
- **Recursive** (variable) carries the whole UI. The `MONO` axis set to 1 is used **only for timecodes** (`03:12`), because timecodes are data. Slight `CASL` warmth on display sizes.
- **Newsreader** is used **only for quoted speech**: transcript lines in citations, and decision/action evidence. Quoted speech is the product's content, so it gets the book face.
- **Scale** (1.2 ratio from 15 px): 12.5 / 15 / 18 / 21.6 / 26 / 31 / 37.
- **Style:** sentence case everywhere, no all-caps eyebrows, no middle-dot meta strings.

**Shape.**
- Radius 3 px for chips and lane blocks, 6 px for panels, round only for the talkback (push-to-talk) button.
- One shadow token, used only for floating layers (ask drawer, command palette).

## Layout

```
┌──────┬───────────────────────────────────────────────────────────┐
│ rail │  Meeting title ● REC 04:12                     [End]      │
│      │  ┌─ lanes ─────────────────────────────────────────────┐  │
│ Mtgs │  │ Priya  ▇▇▇   ▇▇▇▇▇        ▇▇                         │  │
│ Commit│ │ Arjun      ▇▇      ▇▇▇▇▇▇      ▇▇▇                   │  │
│ Decis │ │ Meera           ▇▇            ▇▇▇▇▇  |playhead       │  │
│ People│ └──────────────────────────────────────────────────────┘  │
│ Graph │  transcript (left, 62ch max)      │ extracted by Gemma 4  │
│       │  ▌Priya 03:12  "We ship on the…"   │  Decisions            │
│       │  ▌Arjun 03:20  "I'll own pricing"  │  Action items [ ]     │
│  (●)  │                                    │  Topics               │
└──────┴───────────────────────────────────────────────────────────┘
  (●) talkback button opens the Ask drawer from the right edge
```

- **Alignment:** left-aligned throughout, with a ragged right edge. Transcript lines max out at about 62 characters.
- **Mobile (375 px):**
  - the rail becomes a bottom bar;
  - lanes scroll horizontally inside their panel;
  - the right rail stacks under the transcript;
  - the Ask drawer is full-screen.

## Principles

1. **Speech is evidence.** Every claim the AI makes is shown next to the quote it came from: speaker colour, name, timecode, and a play button.
2. **Show the machine honestly.** Label which model did what: "Heard by Gemma 4 E4B (audio)", "Extracted by Gemma 4". Live graph writes are counted in plain numbers.
3. **Red means live.** The record red appears only while audio is being captured.
4. **Quiet chrome.**
   - No gradients, no glass, no emoji, no sparkle icons.
   - Lucide icons at 1.75 stroke.
   - Motion only in response to an action, plus one deliberate effect: new lane blocks grow in from the playhead.
   - Reduced motion is respected.
