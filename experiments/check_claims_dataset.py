"""
Checks a claims.json against the requested Person 1 improvements.

Run from the repo root (works on an uncommitted local file too):
    python -m experiments.check_claims_dataset data/processed/claims.json

Exit code is 1 if any check FAILS (WARN does not fail).
"""

import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SUPPORTED = {"en", "hi", "bn", "ta", "te", "mr"}
RETRIEVAL_LANGS = {"en", "hi", "bn", "mr"}   # languages Person 2 now retrieves for (update as it grows)
REQUIRED_SAMPLE_KEYS = {"sample_id", "question", "language", "model", "answer", "claims"}
LEGACY_ROOT_KEYS = {"question", "answer", "claims"}

SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)], "mr": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)], "ta": [(0x0B80, 0x0BFF)], "te": [(0x0C00, 0x0C7F)],
}

PRONOUN_START = re.compile(
    r"^\s*((?:he|she|it|they|this|that|these|those|his|her|its|their)\b|"
    r"(?:इसका|इसकी|इसके|इसे|यह|वह|उसका|उसकी|उसके|इन|उन|ये|वे)(?=[\s,।.]|$))", re.IGNORECASE)
TIME_WORDS = re.compile(
    r"\b(current|currently|latest|recent|recently|now|today|present|newest)\b"
    r"|(?<![\u0900-\u097F])(?:वर्तमान|अभी|आज|हाल)(?![\u0900-\u097F])", re.IGNORECASE)   # whole Devanagari words only
VAGUE_WORDS = re.compile(r"\b(numerous|various|many|several|a variety of|a number of|a wide range)\b", re.IGNORECASE)


def script_share(text, lang):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    if lang == "en":
        return sum(c.isascii() for c in letters) / len(letters)
    ranges = SCRIPT_RANGES.get(lang)
    if not ranges:
        return 1.0
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters)


def has_specifics(text):
    return bool(re.search(r"\d", text)) or bool(re.search(r"(?<!^)\b[A-Z][a-z]+", text))


def result(name, status, detail, examples=()):
    print(f"\n[{status}] {name}")
    print(f"       {detail}")
    for e in list(examples)[:4]:
        print(f"         - {e}")
    return status


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/claims.json"
    data = json.load(open(path, encoding="utf-8"))
    samples = data if isinstance(data, list) else data.get("samples", [])
    root = data if isinstance(data, dict) else {}
    statuses = []

    n_claims = sum(len(s.get("claims", [])) for s in samples)
    langs = Counter(s.get("language", "<missing>") for s in samples)
    print(f"File: {path}\nSamples: {len(samples)} | claims: {n_claims} | languages: {dict(langs)}")

    # 1. schema
    missing = [f"sample {s.get('sample_id')}: missing {sorted(REQUIRED_SAMPLE_KEYS - s.keys())}"
               for s in samples if REQUIRED_SAMPLE_KEYS - s.keys()]
    legacy = sorted(LEGACY_ROOT_KEYS & root.keys())
    ids = [s.get("sample_id") for s in samples]
    problems = missing + ([f"legacy top-level keys still present: {legacy}"] if legacy else []) \
        + (["duplicate sample_ids"] if len(ids) != len(set(ids)) else [])
    statuses.append(result("Schema is clean and consistent",
                           "FAIL" if problems else "PASS",
                           f"{len(problems)} problem(s)" if problems else "all samples have the required keys, no legacy root keys",
                           problems))

    # 2. language/script match
    bad = []
    for s in samples:
        lang = s.get("language")
        if lang not in SUPPORTED:
            bad.append(f"sample {s.get('sample_id')}: unsupported language {lang!r}")
            continue
        share = script_share(s.get("answer", ""), lang)
        if share < 0.6:
            bad.append(f"sample {s.get('sample_id')}: labelled {lang} but only {share:.0%} of the answer is in that script")
    statuses.append(result("Language label matches the script of the answer",
                           "FAIL" if bad else "PASS",
                           f"{len(bad)} mismatched sample(s)" if bad else "all answers match their language label", bad))

    # 3. duplicates
    seen, dup_samples = {}, []
    for s in samples:
        key = (s.get("question", "").strip().lower(), s.get("language"))
        if key in seen:
            dup_samples.append(f"sample {s.get('sample_id')} repeats sample {seen[key]} (same question + language)")
        seen.setdefault(key, s.get("sample_id"))
    claim_seen, dup_claims = {}, 0
    for s in samples:
        for c in s.get("claims", []):
            k = c.get("claim", "").strip().casefold()
            if k in claim_seen and claim_seen[k] != s.get("sample_id"):
                dup_claims += 1
            claim_seen.setdefault(k, s.get("sample_id"))
    statuses.append(result("No duplicate samples or repeated claims",
                           "FAIL" if dup_samples else ("WARN" if dup_claims else "PASS"),
                           f"{len(dup_samples)} duplicate sample(s); {dup_claims} claim(s) repeated across samples",
                           dup_samples))

    # 4. self-contained claims
    pron = [f"s{s.get('sample_id')}c{c.get('claim_id')}: {c['claim'][:90]}"
            for s in samples for c in s.get("claims", []) if PRONOUN_START.match(c.get("claim", ""))]
    statuses.append(result("Claims start with a named subject, not a pronoun",
                           "FAIL" if pron else "PASS",
                           f"{len(pron)} of {n_claims} claims start with a pronoun" if pron else "no pronoun-led claims found "
                           "(bare surnames like 'Fleming' still need a manual look)", pron))

    # 5. vague claims
    vague = [f"s{s.get('sample_id')}c{c.get('claim_id')}: {c['claim'][:90]}"
             for s in samples
             for c in s.get("claims", [])
             if script_share(c.get("claim", ""), "en") > 0.8      # English text, even if the language label is missing
             and VAGUE_WORDS.search(c.get("claim", "")) and not has_specifics(c["claim"])]
    statuses.append(result("Vague English claims are filtered out",
                           "WARN" if vague else "PASS", f"{len(vague)} likely-vague claim(s)" if vague else "none found", vague))

    # 6. time-sensitive flag
    ts = [f"s{s.get('sample_id')}c{c.get('claim_id')}: {c['claim'][:90]}"
          for s in samples for c in s.get("claims", [])
          if TIME_WORDS.search(c.get("claim", "")) and not c.get("time_sensitive")]
    statuses.append(result("Time-sensitive claims are flagged (time_sensitive: true)",
                           "WARN" if ts else "PASS",
                           f"{len(ts)} claim(s) with 'current/latest/now' wording but no flag" if ts else "all flagged (or none present)", ts))

    # 7. language scope vs retrieval
    unsup = sorted(l for l in langs if l not in RETRIEVAL_LANGS)
    statuses.append(result("Languages in the dataset are covered by retrieval",
                           "WARN" if unsup else "PASS",
                           f"no retrieval for: {unsup} (those claims get empty evidence)" if unsup else f"all languages covered {sorted(RETRIEVAL_LANGS)}"))

    # 8. size
    uniq = len(claim_seen)
    statuses.append(result("Dataset is big enough to evaluate on",
                           "FAIL" if uniq < 40 else ("WARN" if uniq < 100 else "PASS"),
                           f"{uniq} unique claims (aim for 100+; 40 is the bare minimum for a first look)"))

    c = Counter(statuses)
    print(f"\nSUMMARY: {c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail")
    sys.exit(1 if c["FAIL"] else 0)


if __name__ == "__main__":
    main()
