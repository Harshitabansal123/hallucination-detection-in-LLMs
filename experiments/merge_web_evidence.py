import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
EVIDENCE_FILE = DATA_DIR / "evidence.json"
WEB_FILE = DATA_DIR / "web_evidence.json"
ARCHIVE_DIR = DATA_DIR / "archive"
BACKUP_FILE = ARCHIVE_DIR / "evidence.before_full_merge.json"
TEMP_FILE = EVIDENCE_FILE.with_suffix(".tmp.json")

VAARTA_SOURCES = {
    "vaarta",
    "marathi_wikipedia",
    "sangraha_verified",
    "sangraha_synthetic_deva",
    "sangraha_synthetic_roman",
    "english_wikipedia",
    "hindi_wikipedia",
}


def claim_key(record):
    return record.get("sample_id"), record.get("claim_id")


def local_source_for_language(language):
    if language in {"en", "hi", "bn"}:
        return "miracl"
    if language == "mr":
        return "vaarta"
    return None


def passage_identity(passage):
    """Stable identity for deduplication, preferring docid."""
    docid = passage.get("docid")
    if docid is not None:
        return ("docid", str(docid))
    return (
        "content",
        str(passage.get("title", "")).strip(),
        str(passage.get("text", "")).strip(),
        str(passage.get("url", "")).strip(),
        str(passage.get("source", "")).strip(),
    )


def normalize_local_sources(record):
    language = str(record.get("language") or "").lower()
    default_source = local_source_for_language(language)

    for passage in record.get("evidence", []):
        source = passage.get("source")

        # Preserve explicit original labels such as marathi_wikipedia
        # and sangraha_verified. Only fill missing source labels.
        if not source and default_source:
            passage["source"] = default_source


def add_source_summary(record):
    language = str(record.get("language") or "").lower()
    evidence = record.get("evidence", [])

    counts = Counter(
        str(passage.get("source") or "") for passage in evidence
    )

    miracl_count = counts.get("miracl", 0)
    vaarta_count = sum(
        count for source, count in counts.items()
        if source in VAARTA_SOURCES
    )
    web_count = counts.get("web_search", 0)

    if language == "mr":
        miracl_summary = "not_applicable_here"
        vaarta_summary = vaarta_count
    elif language in {"en", "hi", "bn"}:
        miracl_summary = miracl_count
        vaarta_summary = "not_applicable_here"
    else:
        miracl_summary = "not_applicable_here"
        vaarta_summary = "not_applicable_here"

    unique_passages = {passage_identity(passage) for passage in evidence}

    record["evidence_summary"] = {
        "miracl": miracl_summary,
        "vaarta": vaarta_summary,
        "web_search": web_count,
        "total_evidence": len(unique_passages),
    }


def main():
    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(f"Evidence file not found: {EVIDENCE_FILE}")
    if not WEB_FILE.exists():
        raise FileNotFoundError(f"Web evidence file not found: {WEB_FILE}")

    with EVIDENCE_FILE.open("r", encoding="utf-8") as f:
        local_records = json.load(f)
    with WEB_FILE.open("r", encoding="utf-8") as f:
        web_records = json.load(f)

    if not isinstance(local_records, list) or not isinstance(web_records, list):
        raise ValueError("Both evidence files must contain JSON lists.")

    local_keys = [claim_key(record) for record in local_records]
    if len(local_keys) != len(set(local_keys)):
        raise ValueError("Duplicate (sample_id, claim_id) keys in evidence.json.")

    # Create a new safety backup before the full-file update. Never overwrite
    # an existing backup, so earlier recovery points remain available.
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not BACKUP_FILE.exists():
        shutil.copy2(EVIDENCE_FILE, BACKUP_FILE)

    web_by_key = {}
    for record in web_records:
        key = claim_key(record)
        if key in web_by_key:
            # Combine repeated web records for the same claim instead of
            # silently discarding their passages.
            web_by_key[key].setdefault("evidence", []).extend(
                record.get("evidence", [])
            )
        else:
            web_by_key[key] = {
                **record,
                "evidence": list(record.get("evidence", [])),
            }

    # Deep copy so all transformations happen before the atomic replacement.
    updated_records = json.loads(json.dumps(local_records))
    matched_web_records = 0
    added_web_passages = 0
    deduplicated_passages = 0

    for record in updated_records:
        key = claim_key(record)
        evidence = record.setdefault("evidence", [])

        normalize_local_sources(record)

        # Deduplicate existing passages while retaining the first occurrence.
        unique_evidence = []
        seen = set()
        for passage in evidence:
            identity = passage_identity(passage)
            if identity in seen:
                deduplicated_passages += 1
                continue
            seen.add(identity)
            unique_evidence.append(passage)
        record["evidence"] = evidence = unique_evidence

        web_record = web_by_key.get(key)
        if web_record is not None:
            matched_web_records += 1
            for web_passage in web_record.get("evidence", []):
                item = dict(web_passage)
                item["source"] = "web_search"
                identity = passage_identity(item)

                if identity in seen:
                    deduplicated_passages += 1
                    continue

                evidence.append(item)
                seen.add(identity)
                added_web_passages += 1

        record["retrieval_status"] = (
            "success" if evidence else
            record.get("retrieval_status", "no_evidence_found")
        )
        add_source_summary(record)

    # Confirm all original records remain, and required verifier keys exist.
    required_record_keys = {"sample_id", "claim_id", "claim", "language", "evidence"}
    for record in updated_records:
        missing = required_record_keys - record.keys()
        if missing:
            raise ValueError(
                f"Record {claim_key(record)} is missing required keys: {sorted(missing)}"
            )
        if not isinstance(record["evidence"], list):
            raise ValueError(f"Evidence must be a list for {claim_key(record)}")

    try:
        with TEMP_FILE.open("w", encoding="utf-8") as f:
            json.dump(updated_records, f, ensure_ascii=False, indent=2)
        TEMP_FILE.replace(EVIDENCE_FILE)
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()

    print("FULL MERGE COMPLETED")
    print(f"Records preserved: {len(updated_records)}")
    print(f"Web claim records matched: {matched_web_records}")
    print(f"New web passages added: {added_web_passages}")
    print(f"Duplicate passages skipped: {deduplicated_passages}")
    print(f"Records with evidence_summary: {sum('evidence_summary' in r for r in updated_records)}")
    print(f"Safety backup: {BACKUP_FILE}")
    print(f"Final file: {EVIDENCE_FILE}")

    source_totals = Counter()
    for record in updated_records:
        for passage in record.get("evidence", []):
            source_totals[passage.get("source", "MISSING")] += 1
    print(f"Passage source totals: {dict(source_totals)}")


if __name__ == "__main__":
    main()
