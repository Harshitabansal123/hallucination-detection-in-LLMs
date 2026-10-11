
import json
import re

from src.generation.llm import generate_answer


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}


def get_language_name(language: str) -> str:
    language = language.lower().strip()
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    return LANGUAGE_NAMES[language]


def build_claim_extraction_prompt(
    answer: str,
    language: str,
) -> str:
    language_name = get_language_name(language)

    return f"""
You are a high-recall factual claim extraction system.

The source answer is written in {language_name}.

OBJECTIVE:
Extract all distinct factual propositions explicitly stated in
the source answer. Completeness is more important than brevity.

ATOMICITY:
A claim must express one independently verifiable proposition.
Split sentences containing multiple facts into separate claims.

COVERAGE REQUIREMENTS:
- Extract facts about people, locations, dates, numbers, events,
  relationships, actions, causes, and named entities.
- Include facts expressed in subordinate clauses and lists.
- Preserve every distinct subject-predicate relationship.
- Do not omit a fact merely because another claim is related.
- Do not infer facts that the answer does not explicitly state.
- Preserve uncertainty, qualifications, names, and dates.
- Keep claims in the source language. Do not translate.
- Avoid duplicates and non-factual opinions.
- Do not judge whether the source answer is true.

IMPORTANT EXAMPLE:

Source:
The Taj Mahal is in Agra. It was commissioned by Shah Jahan,
and construction began in 1632.

Required claims:
1. The Taj Mahal is in Agra.
2. The Taj Mahal was commissioned by Shah Jahan.
3. Construction of the Taj Mahal began in 1632.

Before responding, mentally check every sentence and clause
against the claims list. Every distinct fact must be represented.

OUTPUT:
Return only valid JSON in this exact structure:
{{
  "claims": [
    {{"claim_id": 1, "claim": "First claim."}},
    {{"claim_id": 2, "claim": "Second claim."}}
  ]
}}

SOURCE ANSWER:
<answer>
{answer}
</answer>
"""


def build_completeness_prompt(
    answer: str,
    claims: list[dict],
    language: str,
) -> str:
    language_name = get_language_name(language)
    claims_json = json.dumps(
        claims,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are auditing factual claim extraction for completeness.

Both the source answer and existing claims are in {language_name}.

SOURCE ANSWER:
<answer>
{answer}
</answer>

EXISTING CLAIMS:
{claims_json}

TASK:
Find factual propositions explicitly stated in the source answer
that are missing from the existing claims.

Rules:
- Return only missing claims, not claims already represented.
- Split compound statements into atomic propositions.
- Do not add outside knowledge or infer unstated facts.
- Do not rewrite or repeat existing claims.
- Keep the source language unchanged.
- If nothing is missing, return an empty list.

Return only valid JSON:
{{
  "claims": [
    {{"claim_id": 1, "claim": "A missing claim."}}
  ]
}}
"""


def clean_json_output(raw_output: str) -> str:
    cleaned = raw_output.strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No complete JSON object found.")

    return cleaned[start:end + 1]


def parse_claim_json(raw_output: str) -> dict:
    result = json.loads(clean_json_output(raw_output))

    if not isinstance(result, dict):
        raise ValueError("Output must be a JSON object.")

    if not isinstance(result.get("claims"), list):
        raise ValueError("'claims' must be a list.")

    for item in result["claims"]:
        if not isinstance(item, dict):
            raise ValueError("Every claim must be a JSON object.")

        if not isinstance(item.get("claim"), str):
            raise ValueError("Every claim must contain string text.")

    return result


def normalize_claims(result: dict) -> list[dict]:
    final_claims = []
    seen = set()

    for item in result["claims"]:
        claim_text = re.sub(
            r"\s+",
            " ",
            item["claim"],
        ).strip()

        if not claim_text:
            continue

        normalized = claim_text.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        final_claims.append(
            {
                "claim_id": len(final_claims) + 1,
                "claim": claim_text,
            }
        )

    return final_claims


def extract_claims(
    answer: str,
    language: str = "en",
) -> list[dict]:
    if not answer or not answer.strip():
        return []

    language = language.lower().strip()
    get_language_name(language)

    # Pass 1: extract claims from the original answer.
    first_prompt = build_claim_extraction_prompt(
        answer,
        language,
    )

    first_result = None
    last_error = None

    for attempt in range(2):
        prompt = first_prompt

        if attempt == 1:
            prompt += """
Recheck the complete source answer and return valid JSON only.
Make sure the claims field exists and every item has claim text.
"""

        try:
            raw = generate_answer(
                prompt,
                max_new_tokens=700,
                language=language,
            )
            first_result = normalize_claims(
                parse_claim_json(raw)
            )
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc

    if first_result is None:
        raise ValueError(
            f"Initial claim extraction failed: {last_error}"
        )

    # Pass 2: audit the first extraction for missing facts.
    audit_prompt = build_completeness_prompt(
        answer,
        first_result,
        language,
    )

    try:
        audit_raw = generate_answer(
            audit_prompt,
            max_new_tokens=400,
            language=language,
        )

        missing_result = normalize_claims(
            parse_claim_json(audit_raw)
        )

        # Add only claims that are not already represented
        # by an identical normalized string.
        existing = {
            item["claim"].casefold()
            for item in first_result
        }

        for item in missing_result:
            normalized = item["claim"].casefold()

            if normalized not in existing:
                first_result.append(
                    {
                        "claim_id": len(first_result) + 1,
                        "claim": item["claim"],
                    }
                )
                existing.add(normalized)

    except (ValueError, json.JSONDecodeError) as exc:
        # Do not discard the initial extraction if the audit fails.
        print(
            "WARNING: Completeness audit failed; "
            f"returning initial claims. Details: {exc}"
        )

    return first_result
