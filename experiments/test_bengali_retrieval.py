
import json
from pathlib import Path

from src.retrieval.miracl_bengali_shard_retriever import (
    MIRACLBengaliShardRetriever
)
from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path("data/processed/claims.json")
OUTPUT_FILE = Path("data/processed/bengali_evidence_test.json")


def main():
    with open(CLAIMS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    bengali_claims = []

    for sample in data.get("samples", []):
        if sample.get("language") != "bn":
            continue

        for claim in sample.get("claims", []):
            text = claim.get("claim", "").strip()

            if text:
                bengali_claims.append({
                    "sample_id": sample.get("sample_id"),
                    "claim_id": claim.get("claim_id"),
                    "claim": text,
                    "language": "bn"
                })

    print(f"Bengali claims found: {len(bengali_claims)}")

    if not bengali_claims:
        print("No Bengali claims found. Check the language field.")
        return

    retriever = MIRACLBengaliShardRetriever()
    reranker = EvidenceReranker()

    results = []

    for index, item in enumerate(bengali_claims, start=1):
        print(f"[{index}/{len(bengali_claims)}] {item['claim']}")

        candidates = retriever.retrieve(
            item["claim"],
            top_k=50
        )

        evidence = reranker.rerank(
            item["claim"],
            candidates,
            top_k=5
        )

        results.append({
            **item,
            "evidence": evidence,
            "retrieval_status": (
                "success" if evidence else "no_evidence_found"
            )
        })

        print(f"  Candidates: {len(candidates)}")
        print(f"  Final evidence: {len(evidence)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\nBengali retrieval test completed.")
    print(f"Claims processed: {len(results)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
