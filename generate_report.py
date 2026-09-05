"""
Biweekly synthesis job for the AI Value Chain Monitor.

What this does:
1. Calls Claude (with web search enabled) once per value-chain layer (9 layers,
   matching the full stack from energy through applications).
2. Asks it to return ONLY structured JSON: a momentum score, a one-line tag,
   a deployment timeline (today / ~2yr / ~4yr per sub-category), a 2030 bull/bear
   outlook specific to that layer, 5 sourced PUBLIC-company signals -- the most
   important news for that layer over the prior two weeks -- and a PRIVATE-company
   landscape (an overview paragraph plus companies grouped by technology
   sub-category), weighted toward startup/venture press (TechCrunch, The
   Information, Axios Pro Rata, Crunchbase News). Same web search, no extra API
   calls; the model just buckets what it finds by whether the company is
   publicly traded.
3. Assembles all layers plus an overall thesis into report_data.json. (Bull/bear
   used to be a single sector-wide "commentary sentiment" block; it's now per-layer
   and forward-looking to 2030 instead of a snapshot of current commentary.)
4. Writes a timestamped snapshot into /history so you have a version trail.
5. Appends a biweekly digest to newsletter.json: each layer's 5 PUBLIC signals,
   reused as-is, so the Newsletter tab reads as a running "most important news"
   archive per category, linked edition-by-edition, rather than an app/feature
   changelog. Private companies never appear here -- see private_markets.json.
6. Merges each layer's private-company landscape into private_markets.json.
   Merge is additive and never destructive to human review: any company a person
   has flipped to "diligenced": true (or any not-yet-diligenced company already
   present) is left completely untouched no matter what the search finds;
   genuinely new companies are added under a matching (or new) technology
   category as "diligenced": false; a company from a prior cycle that doesn't
   turn up again this cycle is kept, not dropped. Each layer's overview
   paragraph does refresh every cycle, since it's a categorical summary rather
   than a specific company's diligence.

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
                     "data center operators/developers (Equinix, Vantage, QTS, Compass, STACK, Aligned, SpaceX)"},
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
      ["<sub-category / layer name, with example companies in parentheses, COMMA-separated -- e.g. 'Custom ASICs (Google TPU, AWS Trainium, Meta MTIA)', never slash-separated>", "<status deployed TODAY, i.e. roughly this year>", "<status/expectation ~2 years out>", "<status/expectation ~4 years out>"],
      ... 2-4 rows covering the main sub-categories in this layer
    ]
  },
  "outlook2030": {
    "bull": "<2-3 sentences: the bull case for specifically where THIS layer could be by 2030 -- not general AI optimism, argue the case for this layer clearing its current constraint or bottleneck>",
    "bear": "<2-3 sentences: the bear case / key risk for specifically where THIS layer could be by 2030 -- what has to go wrong, slip, or fail to scale for this layer to disappoint>"
  },
  "signals": [
    {"text": "<one sentence, your own words, a specific fact/figure/announcement, about a PUBLICLY TRADED company only>", "source": "<publication name>", "url": "<source url>"},
    ... exactly 5 of these, most recent and most consequential first -- these double as this
    layer's entry in the biweekly news digest (the Newsletter tab), so they should read as
    the 5 most important things that happened in this layer over the last two weeks, not filler
  ],
  "privateLandscape": {
    "overview": "<2-3 sentences: the shape of the PRIVATE company landscape in this layer right now -- what different technology bets/approaches exist among private companies here, and how mature each approach is>",
    "categories": [
      {
        "category": "<a technology sub-category name for grouping private companies in this layer, e.g. 'Fusion' or 'Immersion Cooling'>",
        "companies": [
          {"company": "<name>", "notes": "<one sentence, your own words: what they do, and the most notable recent development if there is one>", "source": "<publication name>", "url": "<source url>"},
          ... every notable PRIVATE (not publicly traded) company you can identify in this sub-category
        ]
      },
      ... 2-4 categories, covering the main technology approaches/sub-segments among PRIVATE companies
      in this layer (they don't need to match the timeline's sub-categories, but may overlap).
      Weight discovery toward startup/venture press specifically -- TechCrunch, The Information,
      Axios Pro Rata, Crunchbase News, PitchBook News -- over generic aggregators.
    ]
  }
}
Rules:
- Use web search to find developments from roughly the last 14 days where possible (this job runs biweekly),
  AND to verify/update any forward-looking timeline dates (commercial operation dates, roadmap milestones) that
  may have shifted since your training data.
- Never quote source text directly; state facts and figures in your own words.
- Prefer primary sources (company filings, earnings calls, government data, official roadmap announcements) and
  reputable outlets over aggregators. Note real uncertainty rather than inventing precision, especially for the
  further-out timeline columns and the 2030 outlook.
- `signals` is PUBLIC companies only, full stop -- never name a private/venture-stage company there, not even
  for a funding round or a named deal. Double check every company in `signals` is actually publicly traded on
  a stock exchange before including it; if genuinely unsure, leave it out of signals rather than guess. Any
  private-company activity, including funding rounds, belongs in `privateLandscape` instead. This keeps the
  two tabs' data cleanly separated.
- Timeline sub-category names (e.g. "Nuclear & SMR (GE Vernova, NuScale, Helion)") may still name private
  companies as examples of who operates in that space -- that's categorical grouping, not a news signal.
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

    # Private-company findings never belong in the public report_data.json --
    # pulled out here and returned separately for the Private Markets merge.
    private_landscape = data.pop("privateLandscape", None) or {"overview": "", "categories": []}

    return data, private_landscape


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


def append_newsletter_entry(edition):
    """Append this cycle's biweekly news digest to newsletter.json.

    The Newsletter tab is a market-news digest, not an app changelog: for each
    layer, the 5 sourced signals already generated for that layer become that
    layer's entry in the digest, dated to this edition. No app/feature updates
    belong here -- those live in commit history / this file's own comments
    instead. Every edition stays in the file so the Newsletter tab can link
    back to all previous editions, not just the latest.
    """
    newsletter_path = "newsletter.json"
    try:
        with open(newsletter_path) as f:
            newsletter = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        newsletter = {"editions": []}

    categories = {
        seg["id"]: [sig["text"] for sig in seg.get("signals", [])]
        for seg in edition["segments"]
    }

    newsletter.setdefault("editions", []).append({
        "date": edition["editionDate"],
        "categories": categories,
    })

    with open(newsletter_path, "w") as f:
        json.dump(newsletter, f, indent=2)


def update_private_markets(private_landscape_by_layer):
    """Merge this cycle's private-company landscape into private_markets.json.

    Each layer is {"overview": str, "categories": [{"category": str, "companies": [...]}]}.

    Additive and non-destructive: a company already present anywhere in a layer
    (in any category, diligenced or not) is left completely untouched -- this
    both protects human diligence and avoids discarding any manual edits made
    to a not-yet-diligenced row. Only genuinely new companies are added, under
    a matching existing category (created if it doesn't exist yet; matched
    case-insensitively so a slightly different label from the model doesn't
    spawn a duplicate category). A company from a prior cycle that doesn't
    resurface this cycle is kept, not dropped -- the landscape only grows
    unless a human removes something. The overview paragraph is refreshed
    every cycle since it's a categorical summary, not a specific company's
    diligence.
    """
    path = "private_markets.json"
    try:
        with open(path) as f:
            pm = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pm = {"layers": {}}

    pm.setdefault("layers", {})
    pm["status"] = "automated landscape (TechCrunch, The Information, etc.) + manual diligence overlay"

    for layer_id, landscape in private_landscape_by_layer.items():
        existing_layer = pm["layers"].get(layer_id) or {"overview": "", "categories": []}
        # Migrate the pre-landscape flat-list format if this is the first run since the upgrade.
        if isinstance(existing_layer, list):
            existing_layer = {"overview": "", "categories": [{"category": "Companies", "companies": existing_layer}]}

        existing_categories = existing_layer.setdefault("categories", [])
        known_names = {
            comp["company"].strip().lower()
            for cat in existing_categories
            for comp in cat.get("companies", [])
        }

        for new_cat in landscape.get("categories", []):
            cat_name = (new_cat.get("category") or "Other").strip()
            target = next(
                (c for c in existing_categories if c["category"].strip().lower() == cat_name.lower()),
                None,
            )
            if target is None:
                target = {"category": cat_name, "companies": []}
                existing_categories.append(target)

            for comp in new_cat.get("companies", []):
                name_key = comp["company"].strip().lower()
                if name_key in known_names:
                    continue  # already present somewhere in this layer -- never touch it
                target["companies"].append({
                    "company": comp["company"],
                    "notes": comp.get("notes", ""),
                    "source": comp.get("source"),
                    "url": comp.get("url"),
                    "diligenced": False,
                })
                known_names.add(name_key)

        if landscape.get("overview"):
            existing_layer["overview"] = landscape["overview"]

        pm["layers"][layer_id] = existing_layer

    pm["lastUpdated"] = datetime.date.today().isoformat()

    with open(path, "w") as f:
        json.dump(pm, f, indent=2)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print("Generating segments...")
    segments = []
    private_landscape_by_layer = {}
    for seg in SEGMENTS:
        print(f"  - {seg['label']}")
        seg_data, private_landscape = generate_segment(client, seg)
        segments.append(seg_data)
        private_landscape_by_layer[seg["id"]] = private_landscape

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

    append_newsletter_entry(edition)
    update_private_markets(private_landscape_by_layer)

    total_private_found = sum(
        len(cat.get("companies", []))
        for landscape in private_landscape_by_layer.values()
        for cat in landscape.get("categories", [])
    )
    print(
        f"Wrote report_data.json, {history_path}, appended to newsletter.json, "
        f"and merged {total_private_found} private-company findings into private_markets.json"
    )


if __name__ == "__main__":
    main()
