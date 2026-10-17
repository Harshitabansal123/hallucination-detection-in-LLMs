import json
import shutil
from pathlib import Path
from datetime import datetime

DATASET = Path("data/processed/claims.json")


def main():
    if not DATASET.exists():
        raise SystemExit(f"Dataset not found: {DATASET.resolve()}")

    data = json.loads(DATASET.read_text(encoding="utf-8"))

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise SystemExit("Unsupported JSON structure.")

    # Repair only records using the new sample_000013-style IDs.
    # Preserve all existing integer IDs and claim text.
    to_repair = [
        sample for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), str)
        and sample["sample_id"].startswith("sample_")
    ]

    if not to_repair:
        print("No sample_XXXXXX IDs found. Nothing to repair.")
        return

    backup = DATASET.with_name(
        f"claims.before_id_fix.{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(DATASET, backup)

    repaired_samples = 0
    repaired_claims = 0

    for sample in to_repair:
        old_id = sample["sample_id"]
        digits = "".join(c for c in old_id if c.isdigit())

        if not digits:
            raise SystemExit(f"Cannot determine numeric ID from {old_id!r}")

        sample["sample_id"] = int(digits)
        repaired_samples += 1

        claims = sample.get("claims", [])
        for index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                raise SystemExit(
                    f"Unexpected claim format in sample {old_id}: {claim!r}"
                )

            old_claim_id = claim.get("claim_id")
            if isinstance(old_claim_id, str):
                claim["claim_id"] = index
                repaired_claims += 1

    # Check that all sample IDs are unique after conversion.
    ids = [s.get("sample_id") for s in samples if isinstance(s, dict)]
    if len(ids) != len(set(ids)):
        shutil.copy2(backup, DATASET)
        raise SystemExit(
            "Duplicate sample IDs detected after conversion. "
            "Original dataset restored from backup."
        )

    temp_path = DATASET.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    temp_path.replace(DATASET)

    print(f"Backup created: {backup}")
    print(f"Repaired sample IDs: {repaired_samples}")
    print(f"Repaired claim IDs: {repaired_claims}")
    print("Claim text and sample answers were not modified.")


if __name__ == "__main__":
    main()