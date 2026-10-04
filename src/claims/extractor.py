import json
import re

from src.generation.llm import generate_answer


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi"
}


def get_language_name(language: str) -> str:

    language = language.lower().strip()

    return LANGUAGE_NAMES.get(
        language,
        language
    )


def build_claim_extraction_prompt(
    answer: str,
    language: str
) -> str:

    language_name = get_language_name(language)

    return f"""
You are an atomic factual claim extraction system.

The original LLM answer is written in {language_name}.

Extract ONLY concrete, independently verifiable factual claims
from the answer.

Rules:

1. Use ONLY information explicitly present in the answer.
2. Do NOT add outside knowledge.
3. Split compound statements into separate claims.
4. Each claim must contain ONE independently verifiable fact.
5. Preserve names, dates, numbers, and qualifiers.
6. Do NOT extract opinions or evaluations.
7. Do NOT extract advice or recommendations.
8. Do NOT extract claims such as "this is important",
   "this is significant", "this is remarkable", etc.
9. Do NOT create duplicate claims.
10. Keep claims concise.
11. Keep claims in the SAME LANGUAGE as the answer.
12. Do NOT translate claims into English.
13. Use standard technical terminology used naturally
    in the target language.
14. Return ONLY valid JSON.
15. Make sure the JSON is COMPLETE and properly closed.

Return exactly this format:

{{
    "claims": [
        {{
            "claim_id": 1,
            "claim": "Atomic factual claim."
        }}
    ]
}}

LLM ANSWER:

{answer}
"""


def clean_json_output(raw_output: str) -> str:

    cleaned = raw_output.strip()

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        cleaned = cleaned[
            start:end + 1
        ]

    return cleaned


def is_valid_claim(claim: str) -> bool:

    claim_lower = claim.lower().strip()

    forbidden_patterns = [

        "significant milestone",
        "important milestone",
        "major milestone",
        "groundbreaking",
        "remarkable achievement",

        "it is important to",
        "it is always a good idea",

        "may change over time",
        "can change over time",

        "should check",
        "good idea to check"
    ]

    for pattern in forbidden_patterns:

        if pattern in claim_lower:
            return False

    return True


def parse_claim_json(
    raw_output: str
) -> dict:

    cleaned_output = clean_json_output(
        raw_output
    )

    return json.loads(
        cleaned_output
    )


def extract_claims(
    answer: str,
    language: str = "en"
) -> list[dict]:

    if not answer or not answer.strip():
        return []

    prompt = build_claim_extraction_prompt(
        answer,
        language
    )

    # First attempt
    raw_output = generate_answer(
        prompt,
        max_new_tokens=500
    )

    try:

        result = parse_claim_json(
            raw_output
        )

    except json.JSONDecodeError:

        print(
            "\nClaim JSON was incomplete."
        )

        print(
            "Retrying claim extraction..."
        )

        # Second attempt with a stronger instruction
        retry_prompt = f"""
Extract the factual claims from the answer below.

IMPORTANT:
- Return ONLY valid JSON.
- Do NOT write explanations.
- Do NOT use Markdown.
- Keep the claims concise.
- Keep the same language as the answer.
- Make sure the JSON is completely closed.
- Generate no more than 8 claims.

Required format:

{{
    "claims": [
        {{
            "claim_id": 1,
            "claim": "factual claim"
        }}
    ]
}}

ANSWER:

{answer}
"""

        raw_output = generate_answer(
            retry_prompt,
            max_new_tokens=500
        )

        try:

            result = parse_claim_json(
                raw_output
            )

        except json.JSONDecodeError as e:

            raise ValueError(
                "Claim extractor returned invalid JSON "
                "after retry.\n\n"
                f"Raw output:\n{raw_output}"
            ) from e


    if "claims" not in result:

        raise ValueError(
            "'claims' field is missing."
        )


    if not isinstance(
        result["claims"],
        list
    ):

        raise ValueError(
            "'claims' must be a list."
        )


    final_claims = []

    seen = set()


    for item in result["claims"]:

        if not isinstance(
            item,
            dict
        ):
            continue


        claim_text = item.get(
            "claim",
            ""
        ).strip()


        if not claim_text:
            continue


        if not is_valid_claim(
            claim_text
        ):
            continue


        normalized = re.sub(
            r"\s+",
            " ",
            claim_text.lower()
        )


        if normalized in seen:
            continue


        seen.add(
            normalized
        )


        final_claims.append(
            {
                "claim_id": len(final_claims) + 1,
                "claim": claim_text
            }
        )


    return final_claims