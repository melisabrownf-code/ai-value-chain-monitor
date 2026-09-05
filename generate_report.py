"""
Biweekly synthesis job for the AI Value Chain Monitor.

What this does:
1. Calls Claude (with web search enabled) once per value-chain layer (9 layers,
   matching the full stack from energy through applications). Before building
   the prompt, reads the company names already on file in private_markets.json
   and market_map.json for that layer and feeds them back in, so the model
   searches for FRESH NEWS on companies it already knows about, not just
   whatever it would have found from a blank slate.
2. Asks it to return ONLY structured JSON: a momentum score, a one-line tag,
   a deployment timeline (today / ~2yr / ~4yr per sub-category), a 2030 bull/bear
   outlook specific to that layer, 5 sourced PUBLIC-company signals -- the most
   important news for that layer over the prior two weeks -- a PRIVATE-company
   landscape, and a PUBLIC-company landscape (each an overview paragraph plus
   companies grouped by technology sub-category). Private discovery is weighted
   toward startup/venture press (TechCrunch, The Information, Axios Pro Rata,
   Crunchbase News). Same web search, no extra API calls; the model just
   buckets what it finds by whether the company is publicly traded.
3. Assembles all layers plus an overall thesis into report_data.json. (Bull/bear
   used to be a single sector-wide "commentary sentiment" block; it's now per-layer
   and forward-looking to 2030 instead of a snapshot of current commentary.)
4. Writes a timestamped snapshot into /history so you have a version trail.
5. Appends a biweekly digest to newsletter.json: each layer's 5 PUBLIC signals,
   reused as-is, so the Newsletter tab reads as a running "most important news"
   archive per category, linked edition-by-edition, rather than an app/feature
   changelog. Private companies never appear here -- see private_markets.json.
6. Merges each layer's private-company landscape into private_markets.json, and
   each layer's public-company landscape into market_map.json -- both landscape
   files use the same merge/refresh logic (merge_landscape_layer()): a company
   flipped to "diligenced": true is a human's own write and is never touched by
   any future cycle; a not-yet-diligenced company already on file gets its
   notes/source/url REPLACED with this cycle's finding (this is the actual
   "search for updates" behavior, driven by step 1 feeding names back in);
   a genuinely new company is added under a matching (or new) category as
   "diligenced": false; a company that doesn't resurface this cycle is left
   exactly as it was. Both files are meant to be hand-edited directly -- that's
   how a human adds a company, corrects one, or marks one diligenced.

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
          ... every notable PRIVATE (not publicly traded) company you can identify in this sub-category --
          both brand-new companies AND fresh updates on any "already tracked" private companies listed below
        ]
      },
      ... 2-4 categories, covering the main technology approaches/sub-segments among PRIVATE companies
      in this layer (they don't need to match the timeline's sub-categories, but may overlap).
      Weight discovery toward startup/venture press specifically -- TechCrunch, The Information,
      Axios Pro Rata, Crunchbase News, PitchBook News -- over generic aggregators.
    ]
  },
  "publicLandscape": {
    "overview": "<2-3 sentences: the shape of the PUBLIC company landscape in this layer right now -- who the leaders are and how the field is organized>",
    "categories": [
      {
        "category": "<a technology or business-model sub-category name for grouping public companies in this layer>",
        "companies": [
          {"company": "<name>", "notes": "<one sentence, your own words: what they do, and the most notable recent development if there is one>", "source": "<publication name>", "url": "<source url>"},
          ... every notable PUBLIC (publicly traded) company you can identify in this sub-category --
          both companies not seen before AND fresh updates on any "already tracked" public companies listed below
        ]
      },
      ... 2-4 categories, covering the main sub-segments among PUBLIC companies in this layer
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
- If the prompt below lists companies already tracked in `privateLandscape` and/or `publicLandscape`, actively
  search for recent news specifically about each of them (not just whatever you'd have found anyway) and
  include an updated entry with fresh notes/source/url if you find something. Still include any genuinely new
  companies you find too. It's fine to return an entry for a tracked company with essentially unchanged notes
  if nothing new turned up -- don't fabricate a development that didn't happen.
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


def tracked_company_names(path, layer_id):
    """Names already on file for this layer, so the prompt can ask for updates
    on them specifically instead of only ever discovering brand-new companies."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    layer = data.get("layers", {}).get(layer_id)
    if not layer or isinstance(layer, list):
        return []
    return [
        comp["company"]
        for cat in layer.get("categories", [])
        for comp in cat.get("companies", [])
    ]


def generate_segment(client, seg):
    system = (
        "You are a research analyst producing one segment of a weekly public-markets "
        "trend report. You track directional momentum, not precise share-level calls. "
        + SEGMENT_SCHEMA_INSTRUCTIONS
    )

    tracked_private = tracked_company_names("private_markets.json", seg["id"])
    tracked_public = tracked_company_names("market_map.json", seg["id"])

    user_content = (
        f"Segment: {seg['label']}.\n"
        f"Focus your web search and synthesis on: {seg['prompt_focus']}.\n"
        f"Today's date: {datetime.date.today().isoformat()}.\n"
    )
    if tracked_private:
        user_content += (
            f"\nPrivate companies already tracked in this layer -- search for recent news on "
            f"each of these specifically, in addition to finding any new ones: {', '.join(tracked_private)}."
        )
    if tracked_public:
        user_content += (
            f"\nPublic companies already tracked in this layer -- search for recent news on "
            f"each of these specifically, in addition to finding any new ones: {', '.join(tracked_public)}."
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

    # Neither landscape belongs in the public report_data.json -- both are
    # pulled out here and returned separately for their own merge/files.
    private_landscape = data.pop("privateLandscape", None) or {"overview": "", "categories": []}
    public_landscape = data.pop("publicLandscape", None) or {"overview": "", "categories": []}

    return data, private_landscape, public_landscape


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


def merge_landscape_layer(existing_layer, fresh_landscape):
    """Merge one cycle's findings for one layer into its on-file landscape.

    existing_layer / fresh_landscape shape: {"overview": str, "categories":
    [{"category": str, "companies": [{"company", "notes", "source", "url",
    "diligenced"}]}]} (fresh_landscape's companies have no "diligenced" key --
    that only exists once a row is on file).

    A company flipped to diligenced=true is a human's own write and is never
    touched by any future cycle, no matter what the search finds -- that's the
    hand-edit contract this whole file is built around. A company already on
    file but NOT diligenced gets its notes/source/url REPLACED with this
    cycle's finding -- this is what makes the biweekly pull an actual refresh,
    since generate_segment() feeds the model its own current name back and
    asks it to search for news on it specifically, not just describe it fresh
    each time from nothing. A genuinely new company is added under a matching
    (or newly created, case-insensitively matched) category as diligenced:false.
    A company that doesn't resurface this cycle is left exactly as it was --
    silence isn't a signal to remove anything.
    """
    existing_layer = existing_layer or {"overview": "", "categories": []}
    existing_categories = existing_layer.setdefault("categories", [])

    index = {}
    for cat in existing_categories:
        for comp in cat.get("companies", []):
            index[comp["company"].strip().lower()] = comp

    for new_cat in fresh_landscape.get("categories", []):
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
            existing_comp = index.get(name_key)

            if existing_comp is not None:
                if existing_comp.get("diligenced"):
                    continue  # a human's own write -- automation never touches it again
                existing_comp["notes"] = comp.get("notes", existing_comp.get("notes", ""))
                existing_comp["source"] = comp.get("source")
                existing_comp["url"] = comp.get("url")
                continue

            new_comp = {
                "company": comp["company"],
                "notes": comp.get("notes", ""),
                "source": comp.get("source"),
                "url": comp.get("url"),
                "diligenced": False,
            }
            target["companies"].append(new_comp)
            index[name_key] = new_comp

    if fresh_landscape.get("overview"):
        existing_layer["overview"] = fresh_landscape["overview"]

    return existing_layer


def update_landscape_file(path, status_text, landscape_by_layer):
    """Merge this cycle's per-layer landscape findings into a landscape file
    (private_markets.json or market_map.json) -- see merge_landscape_layer()
    for the actual per-layer merge/refresh/protect behavior."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"layers": {}}

    data.setdefault("layers", {})
    data["status"] = status_text

    for layer_id, fresh_landscape in landscape_by_layer.items():
        existing_layer = data["layers"].get(layer_id)
        # Migrate the pre-landscape flat-list format if this is the first run since the upgrade.
        if isinstance(existing_layer, list):
            existing_layer = {"overview": "", "categories": [{"category": "Companies", "companies": existing_layer}]}
        data["layers"][layer_id] = merge_landscape_layer(existing_layer, fresh_landscape)

    data["lastUpdated"] = datetime.date.today().isoformat()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print("Generating segments...")
    segments = []
    private_landscape_by_layer = {}
    public_landscape_by_layer = {}
    for seg in SEGMENTS:
        print(f"  - {seg['label']}")
        seg_data, private_landscape, public_landscape = generate_segment(client, seg)
        segments.append(seg_data)
        private_landscape_by_layer[seg["id"]] = private_landscape
        public_landscape_by_layer[seg["id"]] = public_landscape

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
    update_landscape_file(
        "private_markets.json",
        "automated landscape (TechCrunch, The Information, etc.) + manual diligence overlay",
        private_landscape_by_layer,
    )
    update_landscape_file(
        "market_map.json",
        "automated landscape (public filings, earnings calls, market commentary) + manual review overlay",
        public_landscape_by_layer,
    )

    def _count(landscape_by_layer):
        return sum(
            len(cat.get("companies", []))
            for landscape in landscape_by_layer.values()
            for cat in landscape.get("categories", [])
        )

    print(
        f"Wrote report_data.json, {history_path}, appended to newsletter.json, "
        f"merged {_count(private_landscape_by_layer)} private-company findings into private_markets.json, "
        f"and merged {_count(public_landscape_by_layer)} public-company findings into market_map.json"
    )


if __name__ == "__main__":
    main()
