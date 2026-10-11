
import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET_PATH = Path("data/processed/claims.json")


def load_samples(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Support either a list of samples or an object containing "samples".
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])
    else:
        raise ValueError("Dataset must be a JSON list or an object containing a samples list.")

    if not isinstance(samples, list):
        raise ValueError("Dataset must contain a samples list.")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Read-only dataset audit.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the JSON dataset (defaults to data/processed/claims.json)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = load_samples(dataset_path)

    question_groups = defaultdict(list)
    missing_language = []
    missing_claims = []
    missing_answers = []
    missing_review = []
    total_claims = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        question = sample.get("question", "")
        if isinstance(question, str) and question.strip():
            question_groups[question.strip().casefold()].append(sample)

        if not sample.get("language"):
            missing_language.append(sample)

        claims = sample.get("claims")
        if not isinstance(claims, list) or not claims:
            missing_claims.append(sample)
        else:
            total_claims += len(claims)

        answer = sample.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            missing_answers.append(sample)

        if not sample.get("review_status"):
            missing_review.append(sample)

    print("========== DATASET AUDIT ==========")
    print("Dataset:", dataset_path)
    print("Total samples:", len(samples))
    print("Total claims:", total_claims)

    print("\n========== DUPLICATE QUESTIONS ==========")
    duplicate_groups = [
        group for group in question_groups.values() if len(group) > 1
    ]

    if not duplicate_groups:
        print("None found.")
    else:
        for group in duplicate_groups:
            print("\nQuestion:", group[0].get("question"))
            for sample in group:
                print(
                    "  ID:", sample.get("sample_id"),
                    "| Language:", sample.get("language"),
                    "| Claims:", len(sample.get("claims", []))
                    if isinstance(sample.get("claims"), list) else "INVALID",
                )

    print("\n========== MISSING FIELDS ==========")
    print("Missing language:", len(missing_language))
    print("Missing answers:", len(missing_answers))
    print("Missing claims:", len(missing_claims))
    print("Missing review status:", len(missing_review))

    print("\n========== LANGUAGE COUNTS ==========")
    language_counts = defaultdict(int)
    for sample in samples:
        if isinstance(sample, dict):
            language_counts[sample.get("language") or "MISSING"] += 1

    for language, count in sorted(language_counts.items()):
        print(f"{language}: {count}")

    print("\n========== SAMPLES WITH MISSING LANGUAGE ==========")
    for sample in missing_language:
        print("\nSample ID:", sample.get("sample_id"))
        print("Question:", sample.get("question"))
        print("Answer:", sample.get("answer"))
        print("Claims:", sample.get("claims"))

    print("\nAudit completed. No dataset records were modified.")


if __name__ == "__main__":
    main()
