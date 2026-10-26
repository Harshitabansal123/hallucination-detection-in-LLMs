"""
FEVER -> generic evaluation schema adapter.

Converts FEVER claims into:
{
  "id": "...", "claim": "...", "claim_language": "en",
  "evidence": [{"text": "...", "language": "en", "source": "..."}],
  "label": "supported" | "contradicted" | "unsupported"
}
so that experiments/evaluate_verification.py can consume it unchanged.

Label mapping:
    SUPPORTS         -> supported
    REFUTES          -> contradicted
    NOT ENOUGH INFO  -> unsupported

IMPORTANT LIMITS (state these when reporting results)
  * This uses FEVER's GOLD evidence. It does NOT evaluate retrieval.
  * NOT ENOUGH INFO has no gold evidence, so --nei-mode empty gives these
    claims an EMPTY evidence list. The verifier then answers "unsupported"
    automatically, so those rows only test "no evidence" handling, NOT whether
    the NLI model can tell that retrieved evidence is insufficient. They also
    inflate accuracy/macro-F1. Report results with --nei-mode skip as well
    (supported vs contradicted only) for the honest NLI number.
  * Gold evidence sets: each set is a group of sentences that jointly support
    or refute the claim. The verifier scores passages one at a time, so each
    set is concatenated into ONE passage ("Title: sentence Title: sentence").

Inputs
  --claims    FEVER jsonl with labels (e.g. paper_dev.jsonl / paper_test.jsonl /
              shared_task_dev.jsonl). The blind test file has no labels.
  --wiki-dir  directory of FEVER wiki-pages/*.jsonl, needed because the
              claim files only store (page title, sentence id), not the text.
              Not needed if evidence items already carry sentence text.

Run from the repo root:
  python -m experiments.adapters.fever --claims paper_test.jsonl \
      --wiki-dir wiki-pages --out data/eval/fever_test.json --limit 1000 --seed 0
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABEL_MAP = {
    "SUPPORTS": "supported",
    "REFUTES": "contradicted",
    "NOT ENOUGH INFO": "unsupported",
}

_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}


def clean(text):
    for tok, rep in _TOKENS.items():
        text = text.replace(tok, rep)
    return text.strip()


def pretty_title(title):
    return clean(title.replace("_", " "))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_item(item):
    """
    Return (title, sentence_id, text_or_None) for one evidence item, or None.
      original FEVER : [annotation_id, evidence_id, title, sentence_id]
      pre-resolved   : [title, sentence_id, text]
    """
    if not isinstance(item, (list, tuple)):
        return None
    if len(item) >= 4:
        title, sid = item[2], item[3]
        return (title, sid, None) if title is not None and sid is not None else None
    if len(item) == 3 and isinstance(item[0], str):
        return (item[0], item[1], item[2])
    return None


def evidence_sets(claim):
    """
    Normalise a claim's evidence to a list of evidence sets (each a list of items).
      original FEVER   : [ set, set, ... ]  where set = [item, item, ...]
      flat/pre-resolved: [ item, item, ... ] where item = [title, sid, text]
                         -> treated as ONE set
    """
    ev = claim.get("evidence") or []
    if ev and isinstance(ev[0], (list, tuple)) and ev[0] and isinstance(ev[0][0], str):
        return [ev]
    return ev


def needed_lines(claims):
    """{title: set(sentence_ids)} for items whose text is not already present."""
    need = {}
    for c in claims:
        for ev_set in evidence_sets(c):
            for item in ev_set:
                parsed = parse_item(item)
                if parsed and parsed[2] is None:
                    need.setdefault(parsed[0], set()).add(parsed[1])
    return need


def load_wiki_lines(wiki_dir, need):
    """Stream the wiki dump, keeping only the pages/sentences we need."""
    found = {}
    remaining = set(need)
    for path in sorted(Path(wiki_dir).glob("*.jsonl")):
        if not remaining:
            break
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                page = json.loads(line)
                title = page.get("id")
                if title not in remaining:
                    continue
                sents = {}
                for row in (page.get("lines") or "").split("\n"):
                    parts = row.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        sents[int(parts[0])] = parts[1]
                found[title] = sents
                remaining.discard(title)
    return found, remaining


def build_passages(claim, wiki, max_sets):
    passages = []
    for ev_set in evidence_sets(claim)[:max_sets]:
        pieces, sources, seen = [], [], set()
        for item in ev_set:
            parsed = parse_item(item)
            if not parsed:
                continue
            title, sid, text = parsed
            if (title, sid) in seen:
                continue
            seen.add((title, sid))
            if text is None:
                text = wiki.get(title, {}).get(sid)
            if not text or not text.strip():
                continue
            pieces.append(f"{pretty_title(title)}: {clean(text)}")
            sources.append(f"{title}#{sid}")
        if pieces:
            passages.append({
                "text": " ".join(pieces),
                "language": "en",
                "source": "FEVER gold: " + ", ".join(sources),
            })
    return passages


def convert(claims, wiki, nei_mode="empty", max_sets=3):
    out, stats = [], Counter()
    for c in claims:
        raw_label = c.get("label")
        if raw_label not in LABEL_MAP:
            stats["skipped_no_label"] += 1
            continue
        label = LABEL_MAP[raw_label]

        if label == "unsupported":
            if nei_mode == "skip":
                stats["skipped_nei"] += 1
                continue
            evidence = []
        else:
            evidence = build_passages(c, wiki, max_sets)
            if not evidence:
                stats["skipped_unresolved_evidence"] += 1
                continue

        out.append({
            "id": str(c.get("id")),
            "claim": c["claim"],
            "claim_language": "en",
            "evidence": evidence,
            "label": label,
        })
        stats[label] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--wiki-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="random sample size (CPU inference is slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nei-mode", choices=["empty", "skip"], default="empty")
    ap.add_argument("--max-evidence-sets", type=int, default=3)
    args = ap.parse_args()

    claims = read_jsonl(args.claims)
    if args.limit and args.limit < len(claims):
        random.Random(args.seed).shuffle(claims)
        claims = claims[: args.limit]

    need = needed_lines(claims)
    wiki = {}
    if need:
        if not args.wiki_dir:
            raise SystemExit(
                "These claims only store (page, sentence id). "
                "Pass --wiki-dir with the FEVER wiki-pages/*.jsonl files."
            )
        wiki, missing = load_wiki_lines(args.wiki_dir, need)
        if missing:
            print(f"Warning: {len(missing)} pages not found in the wiki dump.")

    converted, stats = convert(claims, wiki, args.nei_mode, args.max_evidence_sets)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(converted)} examples to {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if args.nei_mode == "empty" and stats.get("unsupported"):
        print("Note: NEI rows have EMPTY evidence (trivially 'unsupported'). "
              "Also report a run with --nei-mode skip.")


if __name__ == "__main__":
    main()
