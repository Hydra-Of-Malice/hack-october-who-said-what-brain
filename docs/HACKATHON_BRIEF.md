# Hackathon Brief: Hacktoberfest Hack Day Bengaluru 2026

Source of truth: the official Notion handbook, scraped in full on 15 Oct 2026 into [docs/handbook/](handbook/INDEX.md) (5 pages, 291 blocks, all toggles expanded). Quotes below are verbatim from those files. Anything the handbook does not say is marked **[not specified]**. Findings from outside the handbook (MLH page, MLH rules, ShipWright) are in [TECH_NOTES.md §5](TECH_NOTES.md) and summarised in §9.

## 1. At a glance

| Item | Value | Source |
|---|---|---|
| On-site day | Sunday 18 Oct 2026, 8:30 AM to 6:30 PM IST, Startup Park, Bengaluru | 00-handbook §3 |
| Development window | **13 to 18 Oct 2026**, including the on-site day | 00-handbook §5 |
| Submissions close | **3:30 PM IST, 18 Oct 2026** (tentative; follow organiser updates) | 00-handbook §3, §5 |
| Team size | 1 to 3 | 00-handbook §1, §8 |
| Tracks | Track 1: Best Use of Gemma 4. Track 2: Best Open-Source AI Project | 00-handbook §4 |
| Entries | Exactly **one primary track and one primary problem statement** | 00-handbook §4, FAQ 03 |
| Judging criteria | **Not published in advance** | 00-handbook §5 |
| Host and partners | Hosted by KrowdKraft Community. Powered by MLH and DEV Community. Presenting partner DigitalOcean. Venue partner Startup Park. Innovation partner Where U Elevate. Technology partner ShipWright AI | 00-handbook §1 |

Time left on 15 Oct at about 04:15 IST: roughly **83 hours** until submissions close. On the day itself there are about 5 hours of build time (9:30 AM to 3:30 PM, minus lunch from 12:30 to 1:30).

## 2. Timeline (IST)

| When | What |
|---|---|
| 13 Oct 2026 | Development window opens. ShipWright-qualifying PRs must be opened on or after this date |
| 15 Oct (today) | Phase 1 research and strategy |
| 18 Oct, 8:30 AM | Arrival and individual MLH check-in (QR code at venue) |
| 9:00 to 9:30 AM | Inauguration and orientation |
| 9:30 AM | Build begins |
| 12:30 to 1:30 PM | Lunch |
| 1:30 PM | Building resumes |
| **3:30 PM** | **Project submissions close** |
| 3:45 to 5:30 PM | Project judging |
| 5:30 to 6:00 PM | Final evaluation |
| 6:00 PM | Winner announcements and closing ceremony |
| 6:30 PM | Event concludes |

> "This is the tentative event schedule. Activities and times may change as the Hack Day approaches."

## 3. Team and eligibility rules

- Teams must stay at 1 to 3 members. Members can be added, removed or replaced, but every attending member must independently meet the participation criteria and receive confirmation (FAQ 01).
- One confirmed member may represent the team if the others cannot attend (FAQ 02).
- Shortlisting alone does not guarantee entry. Only participants with an official participation confirmation may attend (§2).
- Each person completes four checks before arriving (§10):
  - an official confirmation email from the organiser address listed in the handbook;
  - an individual registration on the MLH event page;
  - an individual registration on hackFront India 2026, with a team formed or joined there (it may be a different team);
  - their own laptop and charger.
- Walk-ins, on-the-spot team formation and on-the-spot registrations are not permitted (§8).
- Each person checks in individually via MLH using the on-site QR code (§3, §10).
- Lunch is provided. Accommodation and travel are not covered (§10).

## 4. Pre-existing code: what the rules say

Verbatim:

- "Build and demonstrate an **original** AI solution **within the official project development window of 13 to 18 October 2026**." (§1)
- "Teams may build before arriving at the venue, but **only within the stated development window**." (§5)
- "Projects **must be built between 13 and 18 October 2026**." (Track 01 and Track 02 pages)
- "**Submit original work and accurately describe contributions.** Artificial PR activity, misleading submissions or inaccurate participant details may result in disqualification." (§8)
- "Comply with the MLH Code of Conduct, open-source licenses and instructions from the organizers and venue." (§8)

The handbook does not address:

- whether third-party open-source libraries and models may be used **[not specified]**. They clearly can: it is an "open-source AI hackathon", and Track 2 requires open-source or open-weight AI;
- whether a team may reuse **its own** code written before 13 Oct, if disclosed **[not specified]**;
- any required disclosure format **[not specified]**.

**Implication for us.** All three reference repos predate the window:

| Repo | Commit dates |
|---|---|
| who-said-what | 25 Sep to 8 Oct 2026 |
| who-said-what-desktop | 26 Sep to 27 Sep 2026 |
| FalkorDB ("Minutes") | 3 Oct to 12 Oct 2026 |

Copying their code into the submission and presenting it as built during the window would breach "must be built between 13 and 18 October", and presenting it without disclosure would risk "misleading submissions". See [STRATEGY.md](STRATEGY.md) for how we handle this.

## 5. Required sponsor or technology usage

- **Track 1:** "Meaningful use of Gemma 4 must be demonstrated." The track page adds: "Gemma 4 must contribute meaningfully to the solution."
- **Track 2:** "Meaningful use of open-source or open-weight AI must be demonstrated."
- All tracks: "Projects should use the AI technology relevant to their selected track in a meaningful way."
- No other technology is mandatory. **FalkorDB is not mentioned anywhere in the handbook**: it is our choice, not a sponsor requirement. DigitalOcean (presenting partner) is a hosting option, not a requirement.

## 6. Submission checklist

| # | Required item | Notes |
|---|---|---|
| 1 | **Public GitHub repository URL** with the project source code | Create it and keep it public (§5). Must be the same repo used for ShipWright if we enter that challenge |
| 2 | **Working live deployment URL where judges can test the solution** | "You may use Vercel, Netlify, Render or another hosting platform." A laptop-only demo does **not** satisfy this |
| 3 | **Selected track and primary problem statement** | One track, one statement |

- Submission portal: "To be announced."
- Demo video: **[not specified]**.
- Slide deck: **[not specified]**.
- Demo format: judging runs 3:45 to 5:30 PM. "Be prepared to demonstrate how your application works and how it uses the relevant AI capability." Time per team **[not specified]**.

## 7. Judging criteria

> "Judging criteria will not be published in advance. Be prepared to demonstrate how your application works and how it uses the relevant AI capability."

Weights are **[not specified]**. The only stated expectations are:

1. a working application, demonstrable live;
2. visibly *meaningful* use of the track's AI technology;
3. a clear demonstration of the chosen problem statement's approach. The handbook asks teams to "clearly demonstrate the primary chosen approach", even when other capabilities are combined.

MLH's general judging guidance is covered in §9.

## 8. Tracks, problem statements, challenges and prizes

> "These are AI capability approaches, not prescribed solutions and **not eight separate prize categories**. You can build in any industry or application domain."

| Track | ID | Problem statement (handbook wording) |
|---|---|---|
| 1. Best Use of Gemma 4 | G1 | **Autonomous Multi-Agent Collaboration.** Multiple AI agents working together autonomously toward a shared objective; show communication, information exchange, delegated responsibilities and coordinated actions |
| | G2 | **Voice-First AI Interaction.** "Demonstrate how an AI agent understands spoken requests, maintains conversational context and provides meaningful responses or actions" |
| | G3 | **Multimodal AI Intelligence.** Combine text, images or audio; "demonstrate reasoning across multiple input formats" |
| | G5 | **Open Innovation with Gemma 4.** Any solution with "Gemma 4 as a core intelligence component" |
| 2. Best Open-Source AI Project | O1 | **Autonomous Agent Swarms.** Independent agents communicating and coordinating; "meaningful agent-to-agent interaction and collective task execution" |
| | O2 | **Adaptive AI with Memory and Self-Correction.** Retain context, learn from feedback or correct mistakes; "adaptation over time" |
| | O3 | **Trustworthy and Verifiable AI.** Transparency, reliability, accountability; "how users can understand, verify or evaluate information produced by an AI system" |
| | O5 | **Open Innovation with Open-Source AI.** Any solution with meaningful open-source or open-weight AI |

The numbering skips G4 and O4 in the published handbook.

### Prizes

| Prize | Reward |
|---|---|
| Overall Hackathon Winner | Arduino Uno Q board per member (about ₹9,500 each) plus **cash prize TBA** |
| Track 1 winner (Gemma 4) | Arduino Uno Q per member |
| Track 2 winner (Open-Source AI) | Arduino Uno Q per member |
| ShipWright leaderboard | ₹500 Amazon voucher each for the top 3 contributors |
| Social Media Challenge | Official Hacktoberfest merchandise; rules "coming soon" |

Whether one team can win both Overall and a track prize is **[not specified]**.

### ShipWright Leaderboard Challenge (optional; uses the same repo as our submission)

1. Create one public GitHub repo per team.
2. Install the ShipWright GitHub App. One member with admin permission can do this.
3. Open non-draft PRs from feature branches to `main`.
4. Read ShipWright's AI review.
5. Fix valid comments, push and merge.
6. Track progress on the leaderboard.

What qualifies:

- PRs opened **on or after 13 Oct 2026**.
- The PR is created by a human in the team repo (not a fork), reviewed by ShipWright and merged.
- PRs with **more than 5,000 changed lines are skipped**.
- Empty, bot-created, artificial or whitespace-only PRs do not count.
- **Ranking:** PRs with fixes pushed after ShipWright's review rank ahead; the total number of counted PRs breaks ties.
- "A ShipWright suggestion may be incorrect. Implement valid suggestions, explain inapplicable ones and do not merge unsafe changes."

Implication: run the build as many focused PRs of under 5,000 lines each, and respond to each review. This earns leaderboard credit and also produces a timestamped record that the work was done inside the window.

## 9. Grounds for disqualification (stated or directly implied)

- Artificial PR activity (§8).
- Misleading submissions, including misdescribed contributions (§8).
- Inaccurate participant details (§8).
- Implied: work built outside 13 to 18 Oct; breaching the MLH Code of Conduct, open-source licences, or organiser and venue instructions; a team size outside 1 to 3; walk-ins.

### Beyond the handbook

MLH event page, MLH rules on pre-existing work, ShipWright and hackFront details: see [TECH_NOTES.md §5](TECH_NOTES.md). The key lines are pulled into [STRATEGY.md](STRATEGY.md).

## 10. Open questions to put to the organisers (WhatsApp group; a human must send these)

1. May we build on our own open-source code written before 13 Oct if it is clearly disclosed in the README, or must all project code be written from 13 Oct?
2. Can one team win both Overall and a track prize?
3. Does the live deployment need to run every feature (for example live microphone diarization), or is a hosted app over pre-ingested meetings plus an on-site live demo from a laptop acceptable?
4. Is a demo video or deck expected at submission? How long does each team get with the judges?
5. Is the 3:30 PM IST submission deadline final?
