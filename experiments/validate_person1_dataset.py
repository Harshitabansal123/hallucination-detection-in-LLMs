
import json
import sys
from pathlib import Path
from collections import Counter

DEFAULT_DATASET = Path("data/processed/claims_candidates.json")
SUPPORTED_LANGUAGES = {"en", "hi", "bn", "ta", "te", "mr"}


def main():
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET

    if not dataset.exists():
        raise SystemExit(f"Dataset not found: {dataset}")

    try:
        with dataset.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Invalid JSON: {exc}")

    # Accept either a JSON list or the cleaned-file wrapper.
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise SystemExit(
            "ERROR: Expected a JSON list or an object containing a samples list."
        )

    issues = []
    sample_ids = []
    question_counts = Counter()

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            issues.append(f"Record {index}: record is not a JSON object.")
            continue

        sample_id = sample.get("sample_id", "missing")
        prefix = f"Record {index} (ID={sample_id})"

        if not isinstance(sample_id, int) or isinstance(sample_id, bool):
            issues.append(f"{prefix}: sample_id must be an integer.")
        else:
            sample_ids.append(sample_id)

        question = sample.get("question")
        if not isinstance(question, str) or not question.strip():
            issues.append(f"{prefix}: missing or empty question.")
        else:
            question_counts[question.strip().casefold()] += 1

        answer = sample.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            issues.append(f"{prefix}: missing or empty answer.")

        language = sample.get("language")
        if language not in SUPPORTED_LANGUAGES:
            issues.append(f"{prefix}: unsupported or missing language: {language!r}.")

        claims = sample.get("claims")
        if not isinstance(claims, list) or not claims:
            issues.append(f"{prefix}: claims must be a non-empty list.")
            continue

        claim_ids = []
        texts = []

        for claim_index, claim in enumerate(claims, start=1):
            if not isinstance(claim, dict):
                issues.append(
                    f"{prefix}: claim {claim_index} is not a JSON object."
                )
                continue

            claim_id = claim.get("claim_id")
            if not isinstance(claim_id, int) or isinstance(claim_id, bool):
                issues.append(
                    f"{prefix}: claim {claim_index} needs an integer claim_id."
                )
            else:
                claim_ids.append(claim_id)

            claim_text = claim.get("claim")
            if not isinstance(claim_text, str) or not claim_text.strip():
                issues.append(
                    f"{prefix}: claim {claim_index} has missing or empty text."
                )
            else:
                texts.append(claim_text.strip().casefold())

        if len(claim_ids) != len(set(claim_ids)):
            issues.append(f"{prefix}: duplicate claim IDs.")

        if len(texts) != len(set(texts)):
            issues.append(f"{prefix}: exact duplicate claims.")

        # Check that the record remains explicitly marked as a candidate.
        if "dataset_status" in sample and (
            sample["dataset_status"] != "candidate_not_fact_verified"
        ):
            issues.append(f"{prefix}: unexpected dataset_status.")

        if "fact_check_status" in sample and (
            sample["fact_check_status"] != "pending"
        ):
            issues.append(f"{prefix}: unexpected fact_check_status.")

    if len(sample_ids) != len(set(sample_ids)):
        issues.append("Duplicate sample IDs detected.")

    for question, count in question_counts.items():
        if count > 1:
            issues.append(
                f"Repeated question ({count} records): {question!r}. "
                "Review whether these are duplicates or intentional variants."
            )

    print("\nPERSON 1 DATASET VALIDATION")
    print("=" * 55)
    print("Dataset:", dataset)
    print("Total records:", len(samples))
    print("Records with issues:", len(issues))

    if issues:
        print("\nISSUES FOUND:")
        for number, issue in enumerate(issues, start=1):
            print(f"{number}. {issue}")
    else:
        print("\nNo structural issues detected.")

    print("\nIMPORTANT:")
    print("- This checks structure, not factual correctness.")
    print("- It does not modify the dataset.")
    print("- It does not certify claim completeness or language fluency.")


if __name__ == "__main__":
    main()
