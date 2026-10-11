
from src.claims.extractor import extract_claims


TEST_CASES = [
    {
        "name": "Simple fact",
        "answer": "The capital of France is Paris.",
        "expected": [
            "The capital of France is Paris.",
        ],
    },
    {
        "name": "Multiple facts",
        "answer": (
            "The Taj Mahal is in Agra. "
            "It was commissioned by Shah Jahan, "
            "and construction began in 1632."
        ),
        "expected": [
            "The Taj Mahal is in Agra.",
            "The Taj Mahal was commissioned by Shah Jahan.",
            "Construction of the Taj Mahal began in 1632.",
        ],
    },
    {
        "name": "Two distinct facts",
        "answer": (
            "Alexander Fleming discovered penicillin in 1928. "
            "He was a Scottish-born British bacteriologist."
        ),
        "expected": [
            "Alexander Fleming discovered penicillin in 1928.",
            "He was a Scottish-born British bacteriologist.",
        ],
    },
    {
        "name": "Date and location",
        "answer": (
            "India became independent on 15 August 1947. "
            "New Delhi is its capital."
        ),
        "expected": [
            "India became independent on 15 August 1947.",
            "New Delhi is its capital.",
        ],
    },
    {
        "name": "Three facts in one sentence",
        "answer": (
            "Marie Curie was born in Warsaw in 1867 "
            "and won the Nobel Prize in Physics in 1903."
        ),
        "expected": [
            "Marie Curie was born in Warsaw in 1867.",
            "Marie Curie won the Nobel Prize in Physics in 1903.",
        ],
    },
]


def normalize(text):
    return " ".join(text.casefold().split()).strip()


def evaluate_case(case):
    print("\n" + "=" * 70)
    print("TEST:", case["name"])
    print("SOURCE:", case["answer"])

    try:
        claims = extract_claims(case["answer"], language="en")
    except Exception as exc:
        print("\nRESULT: ERROR")
        print(f"{type(exc).__name__}: {exc}")
        return "ERROR"

    actual = [
        item["claim"]
        for item in claims
        if isinstance(item, dict)
        and isinstance(item.get("claim"), str)
    ]

    expected_map = {
        normalize(claim): claim
        for claim in case["expected"]
    }
    actual_map = {
        normalize(claim): claim
        for claim in actual
    }

    missing = [
        claim
        for key, claim in expected_map.items()
        if key not in actual_map
    ]
    extra = [
        claim
        for key, claim in actual_map.items()
        if key not in expected_map
    ]

    print("\nEXPECTED CLAIMS:")
    for claim in case["expected"]:
        print("-", claim)

    print("\nEXTRACTED CLAIMS:")
    for claim in actual:
        print("-", claim)

    if not missing and not extra:
        print("\nRESULT: EXACT MATCH")
        return "PASS"

    print("\nRESULT: MANUAL SEMANTIC REVIEW REQUIRED")

    if missing:
        print("\nEXPECTED CLAIMS NOT MATCHED EXACTLY:")
        for claim in missing:
            print("-", claim)

    if extra:
        print("\nADDITIONAL CLAIMS TO REVIEW:")
        for claim in extra:
            print("-", claim)

    print(
        "\nNote: wording differences can be valid paraphrases. "
        "Review meaning before declaring a failure."
    )
    return "REVIEW"


def main():
    results = {"PASS": 0, "REVIEW": 0, "ERROR": 0}

    for case in TEST_CASES:
        results[evaluate_case(case)] += 1

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("Total cases:", len(TEST_CASES))
    print("Exact matches:", results["PASS"])
    print("Manual review required:", results["REVIEW"])
    print("Errors:", results["ERROR"])
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
