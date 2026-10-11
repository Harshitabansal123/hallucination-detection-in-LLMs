
import json
import os
import tempfile
from pathlib import Path

from src.generation.llm import generate_answer
from src.claims.extractor import extract_claims


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "claims.json"

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}


def get_language_instruction(language: str) -> str:
    language_name = SUPPORTED_LANGUAGES[language]

    return (
        f"Answer entirely in natural, grammatically correct "
        f"{language_name}. "
        "Answer the user's actual question directly. "
        "Do not invent facts. Preserve names, dates, and numbers. "
        "If you are uncertain, state that you are uncertain. "
        "Do not switch languages unnecessarily."
    )


def process_question(
    question: str,
    language: str = "en",
) -> dict:
    """
    Person 1 pipeline:
    Question -> Qwen answer -> atomic claim extraction.

    This pipeline does not verify whether claims are true.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    language = (language or "en").lower().strip()

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    question = question.strip()
    language_instruction = get_language_instruction(language)

    generation_prompt = f"""
{language_instruction}

You are a factual question-answering assistant.

Answer the following question concisely.
Do not add unrelated historical details.
Do not guess when you do not know the answer.

Question:
{question}

Answer:
"""

    print("\nGenerating LLM answer...")

    answer = generate_answer(
        generation_prompt,
        max_new_tokens=300,
        language=language,
    )

    if not answer or not answer.strip():
        raise RuntimeError("The model returned an empty answer.")

    answer = answer.strip()

    # Diagnostic output: inspect exactly what the model generated.
    print("\n========== GENERATION DIAGNOSTICS ==========")
    print("Selected language:", language)
    print("Expected language:", SUPPORTED_LANGUAGES[language])
    print("Raw answer representation:", repr(answer))
    print("============================================")

    print("\nExtracting claims...")

    claims = extract_claims(
        answer,
        language=language,
    )

    # This is a warning, not an automatic factuality judgment.
    if not claims:
        print(
            "\nWARNING: No claims were extracted. "
            "Review the answer before accepting this sample."
        )

    return {
        "question": question,
        "language": language,
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "answer": answer,
        "claims": claims,
        "review_status": "pending",
    }


def save_result(result: dict) -> Path:
    """
    Append one sample to claims.json without silently replacing
    an existing dataset if its JSON structure is invalid.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        try:
            with OUTPUT_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"{OUTPUT_FILE} is invalid JSON. "
                "No changes were made. Back up and repair the file first."
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Dataset root must be a JSON object. No changes were made."
            )

        if not isinstance(data.get("samples"), list):
            raise ValueError(
                "Dataset must contain a 'samples' list. "
                "No changes were made."
            )
    else:
        data = {"samples": []}

    samples = data["samples"]

    existing_ids = [
        sample.get("sample_id")
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("sample_id"), int)
    ]

    new_result = dict(result)
    new_result["sample_id"] = max(existing_ids, default=0) + 1

    samples.append(new_result)

    # Write a complete temporary file before replacing the original.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=OUTPUT_DIR,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            json.dump(
                data,
                temp_file,
                indent=4,
                ensure_ascii=False,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, OUTPUT_FILE)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return OUTPUT_FILE
