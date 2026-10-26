
import json
import os
import time
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.retrieval.reranker import EvidenceReranker

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_FILE = ROOT / "data" / "processed" / "claims.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "web_evidence.json"
BACKUP_FILE = ROOT / "data" / "processed" / "web_evidence.backup.json"

API_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
FINAL_TOP_K = 5



def load_claims():
    with open(CLAIMS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []
    for sample in data.get("samples", []):
        for claim in sample.get("claims", []):
            claims.append({
                "sample_id": sample.get("sample_id"),
                "claim_id": claim.get("claim_id"),
                "claim": claim.get("claim", ""),
                "language": sample.get("language", "en"),
            })

    return claims


def load_existing():
    if not OUTPUT_FILE.exists():
        return {}

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    return {
        (item["sample_id"], item["claim_id"]): item
        for item in records
    }


def save_records(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    temp_file.replace(OUTPUT_FILE)


def search_claim(session, claim):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing from .env")

    response = session.post(
        API_URL,
        json={
            "api_key": api_key,
            "query": claim,
            "search_depth": "basic",
            "max_results": MAX_RESULTS,
        },
        timeout=45,
    )

    response.raise_for_status()
    return response.json().get("results", [])


def retrieve_evidence(session, item):
    results = search_claim(session, item["claim"])
    evidence = []

    for rank, result in enumerate(results, start=1):
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "").strip()

        if not url or not content:
            continue

        evidence.append({
            "docid": (
                f"web_{item['sample_id']}_"
                f"{item['claim_id']}_{rank}"
            ),
            "title": title,
            "text": content,
            "relevance_score": float(result.get("score", 0.0)),
            "rerank_score": None,
            "rank": rank,
            "source": "web",
            "url": url,
        })

    return evidence


def rerank_evidence(reranker, claim, evidence):
    if not evidence:
        return []

    return reranker.rerank(
        claim=claim,
        evidence=evidence,
        top_k=FINAL_TOP_K,
    )


def main():
    claims = load_claims()
    existing = load_existing()

    # Preserve the original output before the first modification.
    if OUTPUT_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(OUTPUT_FILE, BACKUP_FILE)
        print(f"Backup created: {BACKUP_FILE}")

    print(f"Total claims: {len(claims)}")
    print(f"Existing records: {len(existing)}")

    reranker = EvidenceReranker()
    session = requests.Session()

    try:
        for index, item in enumerate(claims, start=1):
            key = (item["sample_id"], item["claim_id"])
            previous = existing.get(key)

            # Reuse existing evidence; rerank it if scores are missing.
            if previous and previous.get("retrieval_status") == "success":
                evidence = previous.get("evidence", [])

                if evidence and any(
                    passage.get("rerank_score") is None
                    for passage in evidence
                ):
                    try:
                        previous["evidence"] = rerank_evidence(
                            reranker,
                            item["claim"],
                            evidence,
                        )
                        existing[key] = previous
                        save_records(list(existing.values()))
                        print(
                            f"[{index}/{len(claims)}] "
                            f"Reranked saved claim {key}; no API call"
                        )
                    except Exception as error:
                        print(f"Reranking failed for {key}: {error}")

                else:
                    print(f"[{index}/{len(claims)}] Already complete: {key}")

                continue

            record = {
                **item,
                "evidence": [],
                "retrieval_status": "pending",
            }

            try:
                evidence = retrieve_evidence(session, item)
                record["evidence"] = rerank_evidence(
                    reranker,
                    item["claim"],
                    evidence,
                )
                record["retrieval_status"] = "success"

                print(
                    f"[{index}/{len(claims)}] "
                    f"Claim {key}: {len(record['evidence'])} "
                    f"passages retrieved and reranked"
                )

            except requests.RequestException as error:
                record["retrieval_status"] = "error"
                record["error"] = str(error)
                print(f"API request failed for {key}: {error}")

                existing[key] = record
                save_records(list(existing.values()))

                if error.response is not None and (
                    error.response.status_code in (401, 403, 429)
                ):
                    print("Stopping due to an API access or usage limit.")
                    break

            except Exception as error:
                record["retrieval_status"] = "error"
                record["error"] = str(error)
                print(f"Retrieval/reranking failed for {key}: {error}")

            existing[key] = record
            save_records(list(existing.values()))

            time.sleep(0.3)

    finally:
        session.close()

    save_records(list(existing.values()))

    records = list(existing.values())
    successful = sum(
        record.get("retrieval_status") == "success"
        for record in records
    )
    passages = [
        passage
        for record in records
        for passage in record.get("evidence", [])
    ]
    missing_scores = sum(
        passage.get("rerank_score") is None
        for passage in passages
    )

    print("\nWeb retrieval and reranking summary")
    print(f"Saved records: {len(records)}")
    print(f"Successful records: {successful}")
    print(f"Passages: {len(passages)}")
    print(f"Missing rerank scores: {missing_scores}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
