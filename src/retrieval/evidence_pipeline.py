
import json
from pathlib import Path

from src.retrieval.miracl_english_shard_retriever import (
    MIRACLEnglishShardRetriever,
)
from src.retrieval.miracl_hindi_shard_retriever import (
    MIRACLShardRetriever,
)
from src.retrieval.miracl_bengali_shard_retriever import (
    MIRACLBengaliShardRetriever,
)
from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path("data/processed/claims.json")
EVIDENCE_FILE = Path("data/processed/evidence.json")

CANDIDATE_TOP_K = 200
FINAL_TOP_K = 5

SUPPORTED_LANGUAGES = {"en", "hi", "bn"}

LANGUAGE_NAMES = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "bangla": "bn",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}


def load_claims():
    with open(CLAIMS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []

    for sample in data.get("samples", []):
        sample_id = sample.get("sample_id")
        metadata_language = sample.get("language")

        for claim in sample.get("claims", []):
            claim_text = claim.get("claim")

            if not isinstance(claim_text, str) or not claim_text.strip():
                continue

            claims.append({
                "sample_id": sample_id,
                "claim_id": claim.get("claim_id"),
                "claim": claim_text.strip(),
                "language": metadata_language,
            })

    return claims


def detect_language(claim, metadata_language):
    language = str(metadata_language or "").strip().lower()
    language = LANGUAGE_NAMES.get(language, language)

    # Use valid language metadata when it is actually present.
    if language in {"en", "hi", "bn", "mr", "ta", "te"}:
        return language

    # Otherwise, fall back to Unicode script detection.
    for char in claim:
        code_point = ord(char)

        if 0x0980 <= code_point <= 0x09FF:
            return "bn"

        if 0x0900 <= code_point <= 0x097F:
            return "hi"

        if 0x0B80 <= code_point <= 0x0BFF:
            return "ta"

        if 0x0C00 <= code_point <= 0x0C7F:
            return "te"

    return "en"


def save_evidence(evidence):
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(EVIDENCE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            evidence,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nEvidence saved to: {EVIDENCE_FILE}")


def main():
    claims = load_claims()
    print(f"Total claims: {len(claims)}")

    for item in claims:
        item["detected_language"] = detect_language(
            item["claim"],
            item["language"],
        )

    language_groups = {
        language: [
            item for item in claims
            if item["detected_language"] == language
        ]
        for language in ("en", "hi", "bn")
    }

    for language, items in language_groups.items():
        print(f"{language.upper()} claims: {len(items)}")

    unsupported_items = [
        item for item in claims
        if item["detected_language"] not in SUPPORTED_LANGUAGES
    ]

    print(f"Unsupported claims: {len(unsupported_items)}")

    if not claims:
        print("No claims found. Nothing to retrieve.")
        return

    retrievers = {}

    if language_groups["en"]:
        print("\nLoading English retriever...")
        retrievers["en"] = MIRACLEnglishShardRetriever()

    if language_groups["hi"]:
        print("\nLoading Hindi retriever...")
        retrievers["hi"] = MIRACLShardRetriever(language="hi")

    if language_groups["bn"]:
        print("\nLoading Bengali retriever...")
        retrievers["bn"] = MIRACLBengaliShardRetriever()

    reranker = None

    if retrievers:
        print("\nLoading semantic reranker...")
        reranker = EvidenceReranker()

    all_evidence = {}

    # Process each supported language independently.
    for language in ("en", "hi", "bn"):
        items = language_groups[language]

        if not items:
            continue

        retriever = retrievers[language]

        print(f"\n{'=' * 70}")
        print(f"{language.upper()} RETRIEVAL")
        print(f"{'=' * 70}")

        if language == "en":
            candidate_batches = retriever.retrieve_many(
                [item["claim"] for item in items],
                top_k=CANDIDATE_TOP_K,
            )
        else:
            candidate_batches = [
                retriever.retrieve(
                    item["claim"],
                    top_k=CANDIDATE_TOP_K,
                )
                for item in items
            ]

        for item, candidates in zip(items, candidate_batches):
            key = (item["sample_id"], item["claim_id"])

            print(
                f"\nProcessing {language.upper()} "
                f"Sample {item['sample_id']} "
                f"Claim {item['claim_id']}"
            )
            print(f"Claim: {item['claim']}")
            print(f"Candidates retrieved: {len(candidates)}")

            # Do not discard short passages: they may contain
            # direct and useful evidence.
            results = (
                reranker.rerank(
                    item["claim"],
                    candidates,
                    top_k=FINAL_TOP_K,
                )
                if candidates
                else []
            )

            all_evidence[key] = {
                "sample_id": item["sample_id"],
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "language": language,
                "evidence": results,
                "retrieval_status": (
                    "success" if results else "no_evidence_found"
                ),
            }

    # Preserve unsupported claims in the output.
    for item in unsupported_items:
        key = (item["sample_id"], item["claim_id"])

        all_evidence[key] = {
            "sample_id": item["sample_id"],
            "claim_id": item["claim_id"],
            "claim": item["claim"],
            "language": item["detected_language"],
            "evidence": [],
            "retrieval_status": "language_not_supported",
        }

    # Restore original claim order.
    final_evidence = []

    for item in claims:
        key = (item["sample_id"], item["claim_id"])

        if key in all_evidence:
            final_evidence.append(all_evidence[key])
        else:
            final_evidence.append({
                "sample_id": item["sample_id"],
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "language": item["detected_language"],
                "evidence": [],
                "retrieval_status": "retrieval_not_completed",
            })

    save_evidence(final_evidence)

    print(f"\n{'=' * 70}")
    print("EVIDENCE RETRIEVAL COMPLETED")
    print(f"{'=' * 70}")
    print(f"Claims processed: {len(final_evidence)}")
    print(f"English claims: {len(language_groups['en'])}")
    print(f"Hindi claims: {len(language_groups['hi'])}")
    print(f"Bengali claims: {len(language_groups['bn'])}")
    print(f"Unsupported claims: {len(unsupported_items)}")
    print(f"Output: {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
