
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
from src.retrieval.vaarta_marathi_retriever import (
    VaartaMarathiRetriever,
)
from src.retrieval.reranker import EvidenceReranker


CLAIMS_FILE = Path("data/processed/claims.json")
EVIDENCE_FILE = Path("data/processed/evidence.json")
WEB_EVIDENCE_FILE = Path("data/processed/web_evidence.json")

CANDIDATE_TOP_K = 200
FINAL_TOP_K = 5

SUPPORTED_LANGUAGES = {"en", "hi", "bn", "mr"}

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
    with CLAIMS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    claims = []

    for sample in data.get("samples", []):
        for claim in sample.get("claims", []):
            claim_text = claim.get("claim")

            if not isinstance(claim_text, str) or not claim_text.strip():
                continue

            claims.append({
                "sample_id": sample.get("sample_id"),
                "claim_id": claim.get("claim_id"),
                "claim": claim_text.strip(),
                "language": sample.get("language"),
            })

    return claims


def detect_language(claim, metadata_language):
    language = str(metadata_language or "").strip().lower()
    language = LANGUAGE_NAMES.get(language, language)

    scripts = {
        "hi": any(0x0900 <= ord(c) <= 0x097F for c in claim),
        "bn": any(0x0980 <= ord(c) <= 0x09FF for c in claim),
        "ta": any(0x0B80 <= ord(c) <= 0x0BFF for c in claim),
        "te": any(0x0C00 <= ord(c) <= 0x0C7F for c in claim),
    }

    has_latin = any(c.isascii() and c.isalpha() for c in claim)

    if scripts["bn"]:
        return "bn"

    if scripts["ta"]:
        return "ta"

    if scripts["te"]:
        return "te"

    # Devanagari is shared by Hindi and Marathi.
    # Explicit metadata distinguishes the two.
    if scripts["hi"]:
        if language in {"hi", "mr"}:
            return language
        return "und"

    if has_latin:
        return "en"

    if language in SUPPORTED_LANGUAGES:
        return language

    return "und"


def claim_key(item):
    return item.get("sample_id"), item.get("claim_id")


def save_evidence(evidence, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(".tmp.json")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)

    temp_file.replace(output_file)
    print(f"\nEvidence saved to: {output_file}")


def load_web_evidence():
    """Load previously retrieved Web Search evidence, if the cache exists."""
    if not WEB_EVIDENCE_FILE.exists():
        print(f"Web Search cache not found: {WEB_EVIDENCE_FILE}")
        return []

    with WEB_EVIDENCE_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("web_evidence.json must contain a JSON list.")
    return records


def source_family(passage):
    source = str(passage.get("source") or "").strip().lower()
    if source in {"miracl", "miracl_en", "miracl_hi", "miracl_bn"}:
        return "miracl"
    if source in {
        "vaarta", "vaarta_marathi", "marathi_wikipedia",
        "sangraha_verified", "sangraha_synthetic_deva",
        "sangraha_synthetic_roman", "english_wikipedia", "hindi_wikipedia",
    }:
        return "vaarta"
    if source in {"web_search", "web", "websearch"}:
        return "web_search"
    return "other"


def add_evidence_summary(record):
    language = str(record.get("language") or "").lower()
    evidence = record.get("evidence", [])
    counts = {"miracl": 0, "vaarta": 0, "web_search": 0}
    for passage in evidence:
        if isinstance(passage, dict):
            family = source_family(passage)
            if family in counts:
                counts[family] += 1

    record["evidence_summary"] = {
        "miracl": counts["miracl"] if language in {"en", "hi", "bn"} else "not_applicable_here",
        "vaarta": counts["vaarta"] if language == "mr" else "not_applicable_here",
        "web_search": counts["web_search"],
        "total_evidence": len(evidence),
    }


def passage_identity(passage):
    if passage.get("url"):
        return ("url", str(passage.get("url")))
    if passage.get("docid"):
        return ("docid", str(passage.get("docid")))
    return (
        "content",
        str(passage.get("title", "")).strip(),
        str(passage.get("text", "")).strip(),
        str(passage.get("source", "")).strip(),
    )


def merge_web_evidence(final_evidence, claims):
    """Merge cached Web Search evidence while preserving canonical claim metadata.

    Web Search passages are read from data/processed/web_evidence.json. This
    function does not make live web requests; it merges the existing cache.
    """
    web_records = load_web_evidence()
    canonical = {claim_key(item): item for item in claims}
    web_by_key = {}

    for record in web_records:
        key = claim_key(record)
        if key not in canonical:
            continue
        bucket = web_by_key.setdefault(key, [])
        for raw_passage in record.get("evidence", []):
            if not isinstance(raw_passage, dict):
                continue
            passage = dict(raw_passage)
            passage["source"] = "web_search"
            bucket.append(passage)

    # Normalize cache metadata using claims.json, while retaining unrelated
    # cached records so a sample-only run cannot erase the other web records.
    normalized_by_key = {}
    for key, passages in web_by_key.items():
        item = canonical[key]
        seen = set()
        unique = []
        for passage in passages:
            identity = passage_identity(passage)
            if identity in seen:
                continue
            seen.add(identity)
            passage["rank"] = len(unique) + 1
            unique.append(passage)
            if len(unique) >= FINAL_TOP_K:
                break
        normalized_by_key[key] = {
            "sample_id": item["sample_id"],
            "claim_id": item["claim_id"],
            "claim": item["claim"],
            "language": item.get("detected_language", item.get("language", "und")),
            "evidence": unique,
            "retrieval_status": "success" if unique else "no_evidence_found",
        }

    # Save canonicalized records for claims in this run and preserve cache rows
    # not touched by this run. Never replace the full cache with a sample subset.
    normalized_web = []
    written_keys = set()
    for item in claims:
        key = claim_key(item)
        if key in normalized_by_key:
            normalized_web.append(normalized_by_key[key])
            written_keys.add(key)
    for record in web_records:
        key = claim_key(record)
        if key not in written_keys and key not in canonical:
            normalized_web.append(record)
    save_evidence(normalized_web, WEB_EVIDENCE_FILE)

    web_lookup = {key: record["evidence"] for key, record in normalized_by_key.items()}
    merged_claims = 0
    for record in final_evidence:
        key = claim_key(record)
        # Canonicalize claim and language even when an old cache has corrupted
        # text or an outdated language such as 'und'.
        item = canonical.get(key)
        if item:
            record["claim"] = item["claim"]
            record["language"] = item.get("detected_language", item.get("language", "und"))

        local_passages = []
        seen = set()
        for raw_passage in record.get("evidence", []):
            if not isinstance(raw_passage, dict):
                continue
            passage = dict(raw_passage)
            family = source_family(passage)
            if family == "web_search":
                continue
            # Fill source only when absent; do not overwrite dataset provenance.
            if not passage.get("source"):
                language = str(record.get("language") or "").lower()
                passage["source"] = "vaarta" if language == "mr" else "miracl"
            identity = passage_identity(passage)
            if identity in seen:
                continue
            seen.add(identity)
            local_passages.append(passage)

        web_passages = []
        for raw_passage in web_lookup.get(key, []):
            passage = dict(raw_passage)
            identity = passage_identity(passage)
            if identity in seen:
                continue
            seen.add(identity)
            web_passages.append(passage)

        # Keep ranks per source as in the requested example: local source ranks
        # 1..5 and Web Search ranks 1..5, rather than renumbering 1..10.
        for rank, passage in enumerate(local_passages, start=1):
            passage["rank"] = rank
        for rank, passage in enumerate(web_passages, start=1):
            passage["rank"] = rank

        record["evidence"] = local_passages + web_passages
        if record["evidence"]:
            record["retrieval_status"] = "success"
        elif record.get("retrieval_status") not in {"language_not_supported"}:
            record["retrieval_status"] = "no_evidence_found"
        add_evidence_summary(record)
        if web_passages:
            merged_claims += 1

    print(f"Web Search cache merged for {merged_claims} claim(s).")
    return final_evidence


def main(output_file=EVIDENCE_FILE, sample_id=None):
    claims = load_claims()

    if sample_id is not None:
        claims = [item for item in claims if item["sample_id"] == sample_id]
        print(f"Claims selected for Sample {sample_id}: {len(claims)}")
    else:
        print(f"Total claims in claims.json: {len(claims)}")

    if not claims:
        print("No claims found. Nothing to retrieve.")
        return

    # Detect language using claim text and sample metadata.
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
        for language in ("en", "hi", "bn", "mr")
    }

    for language, items in language_groups.items():
        print(f"{language.upper()} claims: {len(items)}")

    unsupported_items = [
        item for item in claims
        if item["detected_language"] not in SUPPORTED_LANGUAGES
    ]

    print(f"Unsupported claims: {len(unsupported_items)}")

    # Load existing evidence without discarding unrelated records.
    existing_records = []

    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as file:
            existing_records = json.load(file)

        if not isinstance(existing_records, list):
            raise ValueError(
                "Existing evidence file must contain a JSON list."
            )

    existing_by_key = {}

    for record in existing_records:
        key = claim_key(record)

        if key in existing_by_key:
            raise ValueError(
                f"Duplicate claim ID in evidence file: {key}"
            )

        existing_by_key[key] = record

    # Reuse only when claim text, language, and evidence all match.
    # Corrected claims will be queued for fresh retrieval.
    pending_by_language = {
        language: [] for language in ("en", "hi", "bn", "mr")
    }

    for language, items in language_groups.items():
        for item in items:
            key = claim_key(item)
            existing = existing_by_key.get(key)

            if existing:
                same_claim = (
                    str(existing.get("claim", "")).strip()
                    == item["claim"]
                )
                same_language = existing.get("language") == language
                has_evidence = bool(existing.get("evidence"))

                if same_claim and same_language and has_evidence:
                    print(
                        f"Reusing existing evidence: "
                        f"Sample {item['sample_id']}, "
                        f"Claim {item['claim_id']}"
                    )
                    continue

            pending_by_language[language].append(item)

    pending_total = sum(
        len(items) for items in pending_by_language.values()
    )

    print(f"\nClaims needing retrieval: {pending_total}")

    # Start with existing records. Changed claims are replaced below.
    all_evidence = dict(existing_by_key)

    # Do not send unsupported claims to a guessed language index.
    for item in unsupported_items:
        key = claim_key(item)

        all_evidence[key] = {
            "sample_id": item["sample_id"],
            "claim_id": item["claim_id"],
            "claim": item["claim"],
            "language": item["detected_language"],
            "evidence": [],
            "retrieval_status": "language_not_supported",
        }

        print(
            f"Skipping unsupported language: "
            f"Sample {item['sample_id']}, "
            f"Claim {item['claim_id']}"
        )

    retrievers = {}

    needs_reranker = any(
        pending_by_language[language]
        for language in ("en", "hi", "bn")
    )

    reranker = EvidenceReranker() if needs_reranker else None

    if pending_by_language["en"]:
        print("\nLoading English retriever...")
        retrievers["en"] = MIRACLEnglishShardRetriever()

    if pending_by_language["hi"]:
        print("\nLoading Hindi retriever...")
        retrievers["hi"] = MIRACLShardRetriever(language="hi")

    if pending_by_language["bn"]:
        print("\nLoading Bengali retriever...")
        retrievers["bn"] = MIRACLBengaliShardRetriever()

    if pending_by_language["mr"]:
        print("\nLoading Marathi retriever...")
        retrievers["mr"] = VaartaMarathiRetriever(
            candidate_top_k=CANDIDATE_TOP_K,
            final_top_k=FINAL_TOP_K,
        )

    for language in ("en", "hi", "bn", "mr"):
        items = pending_by_language[language]

        if not items:
            print(f"\n{language.upper()}: no new retrieval needed.")
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

        elif language == "mr":
            # Marathi retriever performs its own semantic reranking.
            candidate_batches = [
                retriever.retrieve(
                    item["claim"],
                    candidate_top_k=CANDIDATE_TOP_K,
                    final_top_k=FINAL_TOP_K,
                )
                for item in items
            ]

        else:
            candidate_batches = [
                retriever.retrieve(
                    item["claim"],
                    top_k=CANDIDATE_TOP_K,
                )
                for item in items
            ]

        if len(candidate_batches) != len(items):
            raise RuntimeError(
                f"{language.upper()} retriever returned "
                f"{len(candidate_batches)} batches for "
                f"{len(items)} claims."
            )

        for item, candidates in zip(items, candidate_batches):
            key = claim_key(item)

            print(
                f"\nProcessing {language.upper()} "
                f"Sample {item['sample_id']} "
                f"Claim {item['claim_id']}"
            )
            print(f"Claim: {item['claim']}")
            print(f"Retrieved candidates: {len(candidates)}")

            if language == "mr":
                results = candidates[:FINAL_TOP_K]

            elif candidates:
                results = reranker.rerank(
                    item["claim"],
                    candidates,
                    top_k=FINAL_TOP_K,
                )

            else:
                results = []

            # Mark retrieval provenance explicitly so evidence_summary can count it.
            for passage in results:
                if isinstance(passage, dict) and not passage.get("source"):
                    passage["source"] = "vaarta" if language == "mr" else "miracl"

            # Replace stale claim text/evidence with this retrieval result.
            # Cached Web Search evidence is merged after retrieval below.
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

    # Write current claims in their original order.
    # Preserve records that are not present in claims.json.
    final_evidence = []
    current_keys = set()

    for item in claims:
        key = claim_key(item)
        current_keys.add(key)

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

    for record in existing_records:
        if claim_key(record) not in current_keys:
            final_evidence.append(record)

    # Merge cached Web Search passages into the combined output and also keep
    # a canonicalized Web Search-only file.
    final_evidence = merge_web_evidence(final_evidence, claims)
    save_evidence(final_evidence, output_file)

    print(f"\n{'=' * 70}")
    print("EVIDENCE RETRIEVAL COMPLETED")
    print(f"{'=' * 70}")
    print(f"Claims loaded: {len(claims)}")
    print(f"Previous records: {len(existing_records)}")
    print(f"Claims needing retrieval: {pending_total}")
    print(f"Final evidence records: {len(final_evidence)}")
    print(f"English claims: {len(language_groups['en'])}")
    print(f"Hindi claims: {len(language_groups['hi'])}")
    print(f"Bengali claims: {len(language_groups['bn'])}")
    print(f"Marathi claims: {len(language_groups['mr'])}")
    print(f"Unsupported claims: {len(unsupported_items)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main(output_file=EVIDENCE_FILE)
