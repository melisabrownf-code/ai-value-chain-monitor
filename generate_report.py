"""
Biweekly synthesis job for the AI Value Chain Monitor.

What this does:
1. Calls Claude (with web search enabled) once per value-chain layer (9 layers,
   matching the full stack from energy through applications).
2. Asks it to return ONLY structured JSON: a momentum score, a one-line tag,
   a deployment timeline (today / ~2yr / ~4yr per sub-category), a 2030 bull/bear
   outlook specific to that layer, and sourced signals.
3. Assembles all layers plus an overall thesis into report_data.json. (Bull/bear
   used to be a single sector-wide "commentary sentiment" block; it's now per-layer
   and forward-looking to 2030 instead of a snapshot of current commentary.)
4. Writes a timestamped snapshot into /history so you have a version trail.

This script is meant to be run on a schedule (see .github/workflows/biweekly-report.yml)
by a runner that has ANTHROPIC_API_KEY set as a secret. It does NOT publish anything
by itself -- report_data.json always lands in a "pending_review" state, and a human
has to flip it to "published" (currently done in report.html's editorial gate; wiring
that button to write back to this same JSON store is the natural next step once the
pilot is validated).
"""

import json
import os
import sys
import datetime
from anthropic import Anthropic

MODEL = "claude-sonnet-5"

SEGMENTS = [
    {"id": "energy", "label": "Energy", "prompt_focus": "renewables (NextEra, Duke, Enphase), nuclear restarts and PPAs, "
                     "small modular reactors (NuScale, Oklo, X-energy, Kairos, GE Vernova), fusion (Helion), "
                     "and grid-scale storage (Form Energy, 4th Power) for AI data centers"},
    {"id": "cooling", "label": "Cooling", "prompt_focus": "air/RDHX cooling (Vertiv, Schneider, Siemens), direct-to-chip "
                     "liquid cooling (JetCool, CoolIT), and immersion cooling (Iceotope, LiquidStack, Submer) adoption"},
    {"id": "power", "label": "Power (Grid/UPS)", "prompt_focus": "grid and UPS equipment (ABB, Eaton, Siemens, Schneider), "
                     "switchgear/transformer lead times, and on-board power delivery silicon (Vicor, AmberSemi, "
                     "backside power delivery nodes)"},
    {"id": "silicon", "label": "Silicon (Fabs/Memory)", "prompt_focus": "leading-edge fabs (TSMC, Samsung, Intel Foundry) "
                     "and memory makers (SK Hynix, Samsung, Micron), process node roadmaps, and HBM supply/demand"},
    {"id": "compute", "label": "Compute (GPU/ASIC)", "prompt_focus": "merchant GPUs (Nvidia, AMD), custom ASICs "
                     "(Google TPU, AWS Trainium, Microsoft Maia, Meta MTIA, Groq), and co-design partners "
                     "(Broadcom, Marvell, EnCharge AI)"},
    {"id": "networking", "label": "Networking", "prompt_focus": "optical transceivers (Lumentum, Coherent, InnoLight), "
                     "co-packaged optics (Lightmatter, Ayar Labs), switches (Arista, Cisco), and "
                     "edge/quantum compute (SEEQC, Cerebras)"},
    {"id": "serversdc", "label": "Servers / DC", "prompt_focus": "OEM server makers (Dell EMC, Supermicro, HPE) and "
                     "data center operators/developers (Equinix, Vantage, QTS, Compass, STACK, Aligned)"},
    {"id": "cloud", "label": "Cloud", "prompt_focus": "hyperscaler cloud capex and results (AWS, Azure, Google Cloud, "
                     "Oracle) and AI-native neoclouds (CoreWeave, Lambda, Crusoe)"},
    {"id": "applications", "label": "Applications", "prompt_focus": "frontier labs (OpenAI, Anthropic, Google DeepMind, "
                     "xAI, Meta AI), enterprise AI tools (Copilot, Glean, Harvey), and consumer AI apps "
                     "(Perplexity, Midjourney)"},
]

SEGMENT_SCHEMA_INSTRUCTIONS = """
Return ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{
  "score": <integer 0-100, momentum: 0=cooling sharply, 50=steady, 100=accelerating fast>,
  "tag": "<one or two words, e.g. 'Accelerating', 'Constrained', 'Steady'>",
  "synthesis": "<2-3 sentences in your own words explaining what's shifting and why it matters, no direct quotes>",
  "timeline": {
    "rows": [
      ["<sub-category / layer name>", "<status deployed TODAY, i.e. roughly this year>", "<status/expectation ~2 years out>", "<status/expectation ~4 years out>"],
      ... 2-4 rows covering the main sub-categories in this layer
    ]
  },
  "outlook2030": {
    "bull": "<2-3 sentences: the bull case for specifically where THIS layer could be by 2030 -- not general AI optimism, argue the case for this layer clearing its current constraint or bottleneck>",
    "bear": "<2-3 sentences: the bear case / key risk for specifically where THIS layer could be by 2030 -- what has to go wrong, slip, or fail to scale for this layer to disappoint>"
  },
  "signals": [
    {"text": "<one sentence, your own words, a specific fact/figure/announcement>", "source": "<publication name>", "url": "<source url>"},
    ... 2 to 3 of these, most recent and most consequential first
  ]
}
Rules:
- Use web search to find developments from roughly the last 14 days where possible (this job runs biweekly),
  AND to verify/update any forward-looking timeline dates (commercial operation dates, roadmap milestones) that
  may have shifted since your training data.
- Never quote source text directly; state facts and figures in your own words.
- Prefer primary sources (company filings, earnings calls, government data, official roadmap announcements) and
  reputable outlets over aggregators. Note real uncertainty rather than inventing precision, especially for the
  further-out timeline columns and the 2030 outlook.
- Public companies only in signals and timeline cells — do not include private/venture-stage companies by name
  unless they are the subject of a specific, sourced, publicly reported deal (e.g. a named PPA or funding round).
- "Trend" tag must be exactly one of: Accelerating, Constrained, Steady.
- outlook2030 must be specific to this layer's own dynamics (its own bottleneck, technology transition, or
  financing structure), not a restatement of generic AI-market bullishness or skepticism.
"""

# Static reference tables don't need re-generation every cycle — the underlying technology
# characteristics don't shift biweekly the way company-specific signals do. Maintained by hand;
# revisit occasionally (e.g. when a new memory type matures or maturity status changes).
STATIC_TABLES = {
    "silicon": [
        {
            "title": "Memory technology comparison (vs. DRAM baseline)",
            "refreshed": "static",
            "trendColumn": False,
            "note": "Technology characteristics, not company-specific — this reference table doesn't need to move every cycle the way the timeline does.",
            "columns": ["Memory type", "Speed", "Density", "Power efficiency", "Endurance", "Cost vs. DRAM", "Maturity"],
            "rows": [
                ["DRAM", "Med", "High", "Med", "Med", "Baseline", "Mainstream / production"],
                ["HBM (stacked DRAM)", "High", "High", "Med", "Med", "Premium", "Mainstream, scaling fast"],
                ["SRAM", "High", "Low", "Low", "Med", "Premium", "Mainstream, confined to on-chip cache"],
                ["MRAM", "Med", "Med", "High", "High", "Premium", "Early testing"],
                ["ReRAM", "High", "High", "High", "Med", "Targeting parity", "Emerging; noise/integration challenges remain"],
                ["PCM (phase-change)", "Med", "Med", "Med", "High", "Targeting parity", "Niche / emerging"],
                ["FeRAM", "Med", "Low", "High", "High", "Targeting parity (higher mfg cost)", "Niche — embedded/industrial use"],
                ["Optical / photonic memory", "High", "Med", "High", "High", "Premium", "Early R&D"],
                ["Memristors / synaptic RAM", "High", "High", "High", "High", "Low (in theory)", "Early R&D — neuromorphic angle"],
            ],
        }
    ]
}

THESIS_SCHEMA_INSTRUCTIONS = """
You will be given the nine segment synopses already generated for this edition. Return ONLY
valid JSON matching this shape:
{
  "thesis": "<3-4 sentences synthesizing the whole value chain this week: what's accelerating, what's constrained, and what the market is actually arguing about right now>"
}
"""


MAX_CLAUDE_ATTEMPTS = 5


def call_claude(client, system, user_content):
    last_error = None
    for attempt in range(1, MAX_CLAUDE_ATTEMPTS + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": user_content}],
        )
        # Concatenate all text blocks (web search may interleave tool_use/tool_result blocks)
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_parts).strip()
        # Defensive cleanup in case the model wraps JSON in a code fence anyway
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.endswith("json"):
                raw = raw[:-4]
        raw = raw.strip()
        try:
            # Parse only the leading JSON object and ignore anything after it --
            # with the web_search tool enabled, the model sometimes appends
            # trailing commentary (e.g. a closing remark or citation note) after
            # an otherwise well-formed JSON object, which a strict json.loads()
            # rejects as "Extra data". Also use strict=False, since the model
            # occasionally emits literal newlines inside a JSON string value
            # (e.g. a multi-line synthesis) instead of escaping them, which
            # Python's default strict JSON parser rejects as an "Invalid control
            # character".
            return json.JSONDecoder(strict=False).raw_decode(raw)[0]
        except json.JSONDecodeError as e:
            last_error = e
            # Empirically, a fraction of web_search-enabled calls come back with
            # no usable text at all (the model spends its whole turn on search
            # round-trips and stop_reason cuts it off before any final text),
            # regardless of how high max_tokens is set. Retrying the same
            # request is cheap relative to losing an entire report run to one
            # unlucky segment.
            print(
                f"    (attempt {attempt}/{MAX_CLAUDE_ATTEMPTS}: "
                f"failed to parse response as JSON -- {e}; retrying)"
                if attempt < MAX_CLAUDE_ATTEMPTS
                else f"    (attempt {attempt}/{MAX_CLAUDE_ATTEMPTS}: giving up -- {e})"
            )
    raise last_error


def generate_segment(client, seg):
    system = (
        "You are a research analyst producing one segment of a weekly public-markets "
        "trend report. You track directional momentum, not precise share-level calls. "
        + SEGMENT_SCHEMA_INSTRUCTIONS
    )
    user_content = (
        f"Segment: {seg['label']}.\n"
        f"Focus your web search and synthesis on: {seg['prompt_focus']}.\n"
        f"Today's date: {datetime.date.today().isoformat()}."
    )
    data = call_claude(client, system, user_content)
    data["id"] = seg["id"]
    data["label"] = seg["label"]

    timeline = data.pop("timeline", None)
    if timeline:
        timeline.setdefault(
            "columns", ["Layer", "2026 — Deployed today", "2028", "2030"]
        )
    data["timeline"] = timeline
    data["tables"] = STATIC_TABLES.get(seg["id"], [])

    return data


def generate_thesis(client, segments):
    system = (
        "You are the editor synthesizing nine segment reports into one weekly thesis "
        "for a public-markets research tool covering the AI value chain. "
        + THESIS_SCHEMA_INSTRUCTIONS
    )
    segment_summary = "\n\n".join(
        f"{s['label']} (score {s['score']}, tag {s['tag']}): {s['synthesis']}"
        for s in segments
    )
    return call_claude(client, system, segment_summary)


def append_log_entry(edition):
    """Append a 'refresh' entry to log.json for this cycle's automated run.

    Feature-level entries (app/UI changes) are written by hand elsewhere; this
    only records that the automated Public Markets pipeline ran and what it
    produced, so the Log tab has a complete history without duplicating what's
    already visible in report_data.json / history/.
    """
    log_path = "log.json"
    try:
        with open(log_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = {"entries": []}

    log.setdefault("entries", []).append({
        "date": edition["editionDate"],
        "type": "refresh",
        "summary": [
            f"Public Markets edition refreshed — {len(edition['segments'])} layers, "
            f"status: {edition['status']}."
        ],
    })

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print("Generating segments...")
    segments = []
    for seg in SEGMENTS:
        print(f"  - {seg['label']}")
        segments.append(generate_segment(client, seg))

    print("Synthesizing overall thesis...")
    thesis_block = generate_thesis(client, segments)

    edition = {
        "editionDate": datetime.date.today().isoformat(),
        "status": "pending_review",  # a human must flip this before it's treated as published
        "thesis": thesis_block["thesis"],
        "segments": segments,
        "generatedAt": datetime.datetime.utcnow().isoformat() + "Z",
    }

    with open("report_data.json", "w") as f:
        json.dump(edition, f, indent=2)

    os.makedirs("history", exist_ok=True)
    history_path = f"history/{edition['editionDate']}.json"
    with open(history_path, "w") as f:
        json.dump(edition, f, indent=2)

    append_log_entry(edition)

    print(f"Wrote report_data.json, {history_path}, and appended to log.json")


if __name__ == "__main__":
    main()
