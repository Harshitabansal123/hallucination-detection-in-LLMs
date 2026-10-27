"""
XNLI -> generic evaluation schema (retrieval-free check of the NLI model).

XNLI gives (premise, hypothesis, label) pairs in many languages, including Hindi. We treat
premise = evidence and hypothesis = claim, so NO retrieval is involved: any error is the
NLI model's own. Labels map to ours:
    entailment -> supported | contradiction -> contradicted | neutral -> unsupported

Run from the repo root (needs:  pip install datasets):
    python -m experiments.adapters.xnli --lang hi --split test --limit 1000 --out data/eval/xnli_hi_test.json
    python -m experiments.evaluate_verification --data data/eval/xnli_hi_test.json

Notes
  * XNLI only covers some of our languages (Hindi yes; Bengali and Marathi are not in XNLI).
  * The pretrained checkpoint was trained on a mixture that includes NLI data in many languages.
    The XNLI test split is meant to be held out, but this adapter cannot check that for you,
    so report XNLI numbers as "sanity check", not as a clean generalisation result.
  * Dataset id: tries "facebook/xnli" first, then the older id "xnli".
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABEL_MAP = {0: "supported", 1: "unsupported", 2: "contradicted"}      # entailment / neutral / contradiction
NAME_MAP = {"entailment": "supported", "neutral": "unsupported", "contradiction": "contradicted"}


def load_rows(lang, split):
    from datasets import load_dataset
    last = None
    for name in ("facebook/xnli", "xnli"):
        try:
            return list(load_dataset(name, lang, split=split))
        except Exception as e:                                               # try the next dataset id
            last = e
    raise SystemExit(f"Could not load XNLI ({lang}, {split}): {last}")


def convert(rows, lang, limit=None, seed=0):
    rows = list(rows)
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    out, skipped = [], Counter()
    for i, r in enumerate(rows):
        premise, hypothesis, lab = (r.get("premise") or "").strip(), (r.get("hypothesis") or "").strip(), r.get("label")
        label = NAME_MAP.get(lab) if isinstance(lab, str) else LABEL_MAP.get(lab)
        if not premise or not hypothesis or label is None:
            skipped["bad_row"] += 1
            continue
        out.append({"id": f"xnli-{lang}-{i}", "claim": hypothesis, "claim_language": lang,
                    "evidence": [{"text": premise, "language": lang, "source": "XNLI premise"}], "label": label})
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out, skipped = convert(load_rows(a.lang, a.split), a.lang, a.limit, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(out)} examples to {a.out}")
    print("  labels:", dict(Counter(x["label"] for x in out)), "| skipped:", dict(skipped) or "none")


if __name__ == "__main__":
    main()
