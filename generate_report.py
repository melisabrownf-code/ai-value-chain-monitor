"""
Weekly synthesis job for the AI Value Chain Monitor.

What this does:
1. Calls Claude (with web search enabled) once per value-chain segment.
2. Asks it to return ONLY structured JSON: a momentum score, a one-line tag,
   a short synthesis paragraph, and 2-3 sourced signals.
3. Assembles all segments plus an overall thesis and bull/bear sentiment
   into report_data.json.
4. Writes a timestamped snapshot into /history so you have a version trail.

This script is meant to be run on a schedule (see .github/workflows/weekly-report.yml)
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
    {
        "id": "semi",
        "label": "Semiconductors (memory & compute)",
        "prompt_focus": "memory chip makers (SK Hynix, Micron, Samsung), HBM supply/demand, "
                         "GPU/ASIC processor announcements (Nvidia, AMD, custom silicon), "
                         "and infrastructure software for AI compute clusters",
    },
    {
        "id": "cooling",
        "label": "Cooling & facility infrastructure",
        "prompt_focus": "data center liquid cooling adoption, rack power density trends, "
                         "and facility design constraints for AI-scale deployments",
    },
    {
        "id": "energy",
        "label": "Energy & power",
        "prompt_focus": "power purchase agreements, nuclear/SMR deals, grid interconnection "
                         "queues, and energy availability as a constraint on AI data center buildout",
    },
    {
        "id": "hyperscale",
        "label": "Hyperscaler capex",
        "prompt_focus": "capital expenditure guidance and results from Microsoft, Amazon, "
                         "Google/Alphabet, Meta and Oracle, and any commentary on free cash flow "
                         "or financing strain tied to that spending",
    },
    {
        "id": "llm",
        "label": "Frontier models / LLM layer",
        "prompt_focus": "compute deals and capacity commitments from frontier AI labs "
                         "(OpenAI, Anthropic, Google DeepMind, xAI), model release cadence, "
                         "and revenue run-rate disclosures",
    },
]

SEGMENT_SCHEMA_INSTRUCTIONS = """
Return ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{
  "score": <integer 0-100, momentum: 0=cooling sharply, 50=steady, 100=accelerating fast>,
  "tag": "<one or two words, e.g. 'Accelerating', 'Steady', 'Constrained', 'Cooling'>",
  "synthesis": "<2-3 sentences in your own words explaining what's shifting and why it matters, no direct quotes>",
  "signals": [
    {"text": "<one sentence, your own words, a specific fact/figure/announcement>", "source": "<publication name>", "url": "<source url>"},
    ... 2 to 3 of these, most recent and most consequential first
  ]
}
Rules:
- Use web search to find developments from roughly the last 7-14 days where possible.
- Never quote source text directly; state facts and figures in your own words.
- Prefer primary sources (company filings, earnings calls, government data) and reputable
  outlets over aggregators. Note real uncertainty rather than inventing precision.
"""

THESIS_SCHEMA_INSTRUCTIONS = """
You will be given the five segment synopses already generated for this edition. Return ONLY
valid JSON matching this shape:
{
  "thesis": "<3-4 sentences synthesizing the whole value chain this week: what's accelerating, what's constrained, and what the market is actually arguing about right now>",
  "sentiment": {
    "position": <integer 0-100, 0 = fully bullish commentary, 100 = fully bearish commentary>,
    "bull": "<2-3 sentences summarizing the strongest bull-case commentary you found, in your own words>",
    "bear": "<2-3 sentences summarizing the strongest bear-case/skeptical commentary you found, in your own words, drawing on independent research, Substack, or Medium-style commentary if available>"
  }
}
"""


def call_claude(client, system, user_content):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
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
    return json.loads(raw)


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
    return data


def generate_thesis(client, segments):
    system = (
        "You are the editor synthesizing five segment reports into one weekly thesis "
        "for a public-markets research tool covering the AI value chain. "
        + THESIS_SCHEMA_INSTRUCTIONS
    )
    segment_summary = "\n\n".join(
        f"{s['label']} (score {s['score']}, tag {s['tag']}): {s['synthesis']}"
        for s in segments
    )
    return call_claude(client, system, segment_summary)


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

    print("Synthesizing overall thesis + sentiment...")
    thesis_block = generate_thesis(client, segments)

    edition = {
        "editionDate": datetime.date.today().isoformat(),
        "status": "pending_review",  # a human must flip this before it's treated as published
        "thesis": thesis_block["thesis"],
        "sentiment": thesis_block["sentiment"],
        "segments": segments,
        "generatedAt": datetime.datetime.utcnow().isoformat() + "Z",
    }

    with open("report_data.json", "w") as f:
        json.dump(edition, f, indent=2)

    os.makedirs("history", exist_ok=True)
    history_path = f"history/{edition['editionDate']}.json"
    with open(history_path, "w") as f:
        json.dump(edition, f, indent=2)

    print(f"Wrote report_data.json and {history_path}")


if __name__ == "__main__":
    main()
