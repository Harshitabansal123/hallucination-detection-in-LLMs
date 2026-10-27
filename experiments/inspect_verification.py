"""
Show WHY the verifier gave each label: per-passage NLI scores plus passage text.

Run from the repo root:
    python -m experiments.inspect_verification            # all claims
    python -m experiments.inspect_verification 1:2 5:3    # only sample:claim pairs
"""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

VERIFICATION_FILE = "data/processed/verification.json"
EVIDENCE_FILE = "data/processed/evidence.json"


def main():
    wanted = set(sys.argv[1:])  # e.g. "1:2" = sample 1, claim 2

    with open(VERIFICATION_FILE, encoding="utf-8") as f:
        verification = json.load(f)
    try:                                   # only needed for older verification.json files
        with open(EVIDENCE_FILE, encoding="utf-8") as f:
            evidence = json.load(f)
    except FileNotFoundError:
        evidence = []

    # passage text lives in evidence.json (verification.json only keeps scores)
    text_of = {}
    for item in evidence:
        for p in item["evidence"]:
            text_of[(item["sample_id"], item["claim_id"], p["docid"])] = p["text"]

    for r in verification:
        key = f"{r['sample_id']}:{r['claim_id']}"
        if wanted and key not in wanted:
            continue

        print("=" * 90)
        print(f"sample {r['sample_id']} claim {r['claim_id']} ({r['language']})  ->  "
              f"{r['label'].upper()}   support={r['support_score']:.2f}  contradict={r['contradict_score']:.2f}")
        print("CLAIM:", r["claim"])

        for ps in r["passage_scores"]:
            text = ps.get("text_preview") or text_of.get((r["sample_id"], r["claim_id"], ps["docid"]), "")
            text = text.replace("\n", " ")[:170]
            tag = f" [{ps['source_type']}" + (f" chunk {ps['chunk']}" if ps.get("chunk") else "") + "]" if ps.get("source_type") else ""
            print(f"  rank {ps['rank']}  ent {ps['entailment']:.2f}  neu {ps['neutral']:.2f}  "
                  f"con {ps['contradiction']:.2f}  | {ps['title'][:60]}{tag}")
            print(f"        {text}")


if __name__ == "__main__":
    main()
