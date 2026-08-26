# AI Value Chain Monitor — Pilot

## What's in this folder
- `report.html` — the front end. Open it directly; it's fully self-contained (no server needed) and seeded with this week's real edition.
- `generate_report.py` — the backend job. Calls Claude with web search once per segment, returns structured JSON, and writes `report_data.json` + a dated snapshot in `/history`.
- `.github/workflows/weekly-report.yml` — runs `generate_report.py` every Monday and commits the refreshed `report_data.json` back to the repo.
- `requirements.txt` — the one dependency (`anthropic`).

## To wire up the weekly cadence
1. Push this folder to a GitHub repo.
2. In the repo's Settings → Secrets and variables → Actions, add a secret named `ANTHROPIC_API_KEY` with your API key.
3. That's it — the workflow runs every Monday at 13:00 UTC, or you can trigger it manually from the Actions tab (`workflow_dispatch`).
4. Every run writes a fresh `report_data.json` in `pending_review` status, plus a dated copy in `/history` so you have a full version trail.
5. Host `report.html` wherever you like (GitHub Pages is the easiest zero-cost option) and point it at `report_data.json` in the same repo — right now it uses embedded demo data, so the one code change left is swapping the embedded `SEGMENTS` array for a `fetch('report_data.json')` call.

## The editorial gate
`report_data.json` always comes out of the automated job as `pending_review` — nothing is auto-published. The pilot's approve/reset buttons in `report.html` currently write that status to the artifact's own storage as a stand-in; connecting that button to actually update `report_data.json` (e.g. via a small API endpoint or a second GitHub Action triggered by a repo dispatch event) is the natural next step once you're happy with synthesis quality.

## Validating the pilot
Before expanding to more reports, the things worth checking over the next few weekly cycles:
- Are the momentum scores/tags directionally right, or does the model over/under-react to noisy weeks?
- Are the sourced signals actually the most consequential ones, or is it picking up SEO filler?
- Does the bull/bear sentiment section reflect real commentary, or drift toward blandness?
