
import json
from pathlib import Path
from collections import Counter

SOURCE = Path("data/processed/claims_review.json")
OUTPUT = Path("data/processed/claims_candidates.json")

# Initial candidates based on the review we have performed.
# Sample 1 is included with a corrected language label in the new copy.
# Samples 2, 3, and 10 are English candidates, not fact-verified records.
INCLUDE_IDS = {1, 2, 3, 10}


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Source file not found: {SOURCE}")

    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists. Nothing was overwritten. "
            "Inspect the existing file before deciding what to do."
        )

    with SOURCE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", [])

    selected = []

    for sample in samples:
        sample_id = sample.get("sample_id")

        if sample_id not in INCLUDE_IDS:
            continue

        item = dict(sample)

        if sample_id == 1:
            item["language"] = "en"

        item["dataset_status"] = "candidate_not_fact_verified"
        item["fact_check_status"] = "pending"

        selected.append(item)

    if not selected:
        raise SystemExit("No candidate samples were selected.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print("Candidate dataset created:", OUTPUT)
    print("Candidate count:", len(selected))
    print("Included sample IDs:", [s.get("sample_id") for s in selected])
    print("Language counts:", dict(Counter(
        s.get("language", "missing") for s in selected
    )))
    print("Fact-check status: pending for every candidate")
    print("Original claims.json and claims_review.json were not modified.")


if __name__ == "__main__":
    main()
