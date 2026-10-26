"""
Person 2 -> Person 3 -> Person 4 connector.

Reads  data/processed/evidence.json      (Person 2's output)
Writes data/processed/verification.json  (input for Person 4)

Run from the repo root:
    python -m src.verification.pipeline

Person 2's format (one item per claim):
{
  "sample_id", "claim_id", "claim", "language", "retrieval_status", "evidence_summary",
  "evidence": [{"docid", "title", "text", "rank", "relevance_score", "rerank_score",
                "source", "language"?, "url"?}, ...]
}
Older files (no source/url/retrieval_status) still work.

What this connector does
  * Prepends the passage title ("Title: text") for context.
  * Splits LONG passages (web pages can be 8,000 characters) into sentence-based
    chunks of at most MAX_CHARS, because the NLI model only reads ~512 tokens and
    would otherwise never see the relevant sentence. Each chunk is scored on its own
    and the strongest entailment / contradiction over all chunks decides the label.
  * Keeps the source type (miracl / web_search / ...) so results can be reported per source.

verifier.py is NOT modified: this file uses its public nli_scores() and aggregate().
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.verification.verifier import nli_scores, aggregate

EVIDENCE_FILE = Path("data/processed/evidence.json")
VERIFICATION_FILE = Path("data/processed/verification.json")

MAX_CHARS = 800      # per chunk; Indic scripts use many tokens per character
MAX_CHUNKS = 8       # per passage, bounds CPU time on huge web pages
PREVIEW_CHARS = 200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n+")


def load_evidence(path=EVIDENCE_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text, max_chars=MAX_CHARS, max_chunks=MAX_CHUNKS):
    """Split text into sentence-based chunks of at most max_chars characters."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:          # a single very long "sentence"
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current)
    return chunks[:max_chunks]


def to_verifier_passages(item):
    """Convert Person 2's passages into the verifier's evidence format (one entry per chunk)."""
    passages = []
    for p in item.get("evidence", []):
        title = (p.get("title") or "").strip()
        source_type = p.get("source") or "miracl"
        language = p.get("language") or item.get("language")
        for i, chunk in enumerate(chunk_text(p.get("text"))):
            passages.append({
                "text": f"{title}: {chunk}" if title else chunk,
                "language": language,
                "source": f"{source_type}:{p.get('docid')}",
                "source_type": source_type,
                "docid": p.get("docid"),
                "title": title,
                "url": p.get("url"),
                "rank": p.get("rank"),
                "chunk": i,
                "rerank_score": p.get("rerank_score"),
                "relevance_score": p.get("relevance_score"),
            })
    return passages


def verify_item(item, scorer=nli_scores):
    """Verify one claim against its retrieved passages."""
    passage_results = []
    for passage in to_verifier_passages(item):
        passage_results.append(
            {"evidence": passage, "scores": scorer(passage["text"], item["claim"])}
        )

    decision = aggregate(passage_results)

    return {
        "sample_id": item.get("sample_id"),
        "claim_id": item.get("claim_id"),
        "claim": item["claim"],
        "language": item.get("language"),
        "retrieval_status": item.get("retrieval_status"),
        "label": decision["label"],
        "support_score": decision["support_score"],
        "contradict_score": decision["contradict_score"],
        "best_evidence": decision["best_evidence"],
        "reason": decision.get("reason"),
        "num_passages": len(item.get("evidence", [])),
        "num_chunks_scored": len(passage_results),
        "passage_scores": [
            {
                "docid": r["evidence"]["docid"],
                "chunk": r["evidence"]["chunk"],
                "source_type": r["evidence"]["source_type"],
                "rank": r["evidence"]["rank"],
                "title": r["evidence"]["title"],
                "url": r["evidence"]["url"],
                "text_preview": r["evidence"]["text"][:PREVIEW_CHARS],
                **r["scores"],
            }
            for r in passage_results
        ],
    }


def run(items, scorer=nli_scores, verbose=False):
    results = []
    for i, item in enumerate(items, start=1):
        res = verify_item(item, scorer)
        results.append(res)
        if verbose:
            print(f"[{i}/{len(items)}] sample {res['sample_id']} claim {res['claim_id']} "
                  f"({res['language']}): {res['label']} "
                  f"(sup {res['support_score']:.2f}, con {res['contradict_score']:.2f}, "
                  f"{res['num_chunks_scored']} chunks)")
    return results


def save_verification(results, path=VERIFICATION_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nVerification saved to: {path}")


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Verify retrieved evidence with the NLI verifier.")
    ap.add_argument("--evidence", default=str(EVIDENCE_FILE), help="input: Person 2's evidence JSON")
    ap.add_argument("--out", default=str(VERIFICATION_FILE), help="output: verification JSON (never overwritten unless you point --out at it)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    items = load_evidence(args.evidence)
    print(f"Claims to verify: {len(items)}  (evidence: {args.evidence} -> {args.out})")

    results = run(items, verbose=True)
    save_verification(results, args.out)

    print("\nLabel counts:", dict(Counter(r["label"] for r in results)))
    print("By language :", dict(Counter((r["language"], r["label"]) for r in results)))


if __name__ == "__main__":
    main()
