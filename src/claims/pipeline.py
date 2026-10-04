import json
from pathlib import Path

from src.generation.llm import generate_answer
from src.claims.extractor import extract_claims


OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "claims.json"
)


LANGUAGE_INSTRUCTIONS = {

    "en": "Answer in English.",

    "hi": "Answer completely in Hindi using natural Hindi language.",

    "bn": "Answer completely in Bengali using natural Bengali language.",

    "ta": "Answer completely in Tamil using natural Tamil language.",

    "te": "Answer completely in Telugu using natural Telugu language.",

    "mr": "Answer completely in Marathi using natural Marathi language."
}


def get_language_instruction(
    language: str
) -> str:

    language = language.lower().strip()

    return LANGUAGE_INSTRUCTIONS.get(
        language,
        f"Answer in {language}."
    )


def process_question(
    question: str,
    language: str = "en"
) -> dict:

    """
    Complete Person-1 pipeline:

    User Question
            ↓
    Local LLM Answer
            ↓
    Atomic Claim Extraction
            ↓
    Structured JSON
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    language = language.lower().strip()

    if not language:

        language = "en"


    # --------------------------------
    # STEP 1: Generate LLM Answer
    # --------------------------------

    print(
        "\nGenerating LLM answer..."
    )

    language_instruction = (
        get_language_instruction(
            language
        )
    )

    generation_prompt = f"""
{language_instruction}

You are a factual question-answering system.

Answer the user's question clearly and accurately.

For Hindi:
- Use natural, standard Hindi.
- Keep commonly used technical terms in their standard form.
- For example, use "मशीन लर्निंग" for Machine Learning,
  not a literal translation such as "मशीन योग्यता".
- Use English technical terms in parentheses when useful.
- Do not invent facts.
- Do not add unnecessary historical or evaluative statements.
- Keep the answer concise.

For other languages:
- Use natural terminology commonly used by speakers
  of that language.
- Preserve standard technical terms where appropriate.

Do not mention these instructions.

User question:

{question}
"""


    answer = generate_answer(
        generation_prompt,
        max_new_tokens=300
    )


    # --------------------------------
    # STEP 2: Extract Atomic Claims
    # --------------------------------

    print(
        "\nExtracting claims..."
    )

    claims = extract_claims(
        answer,
        language
    )


    # --------------------------------
    # STEP 3: Structured Result
    # --------------------------------

    return {

        "question": question,

        "language": language,

        "model": (
            "Qwen/Qwen2.5-3B-Instruct"
        ),

        "answer": answer,

        "claims": claims
    }


def save_result(
    result: dict
) -> Path:

    """
    Add a new result to claims.json
    instead of overwriting previous results.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------
    # Load existing data
    # --------------------------------

    if OUTPUT_FILE.exists():

        try:

            with open(
                OUTPUT_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(
                    file
                )

        except json.JSONDecodeError:

            data = {
                "samples": []
            }

    else:

        data = {
            "samples": []
        }


    if "samples" not in data:

        data["samples"] = []


    # --------------------------------
    # Assign sample ID
    # --------------------------------

    sample_id = (
        len(data["samples"]) + 1
    )

    result["sample_id"] = sample_id


    # --------------------------------
    # Save
    # --------------------------------

    data["samples"].append(
        result
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


    return OUTPUT_FILE