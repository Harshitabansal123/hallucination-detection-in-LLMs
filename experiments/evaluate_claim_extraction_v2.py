
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

    try:
        extracted = extract_claims(case["answer"], language="en")
    except Exception as exc:
        print("ERROR:", type(exc).__name__, str(exc))
        return None

    actual = list(dict.fromkeys(
        normalize(item["claim"])
        for item in extracted
        if isinstance(item, dict)
        and isinstance(item.get("claim"), str)
        and item["claim"].strip()
    ))

    expected = list(dict.fromkeys(
        normalize(claim) for claim in case["expected"]
    ))

    expected_set = set(expected)
    actual_set = set(actual)

    true_positive = len(expected_set & actual_set)
    false_positive = len(actual_set - expected_set)
    false_negative = len(expected_set - actual_set)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("Reference claims:", len(expected))
    print("Extracted claims:", len(actual))
    print("Exact matches:", true_positive)
    print("Unmatched extracted claims:", false_positive)
    print("Reference claims missed:", false_negative)
    print(f"Exact-match precision: {precision:.3f}")
    print(f"Exact-match recall:    {recall:.3f}")
    print(f"Exact-match F1:        {f1:.3f}")

    if false_positive or false_negative:
        print("Status: MANUAL SEMANTIC REVIEW REQUIRED")
        if false_negative:
            print("Unmatched reference claims:")
            for claim in expected_set - actual_set:
                print("-", claim)
        if false_positive:
            print("Unmatched extracted claims:")
            for claim in actual_set - expected_set:
                print("-", claim)
    else:
        print("Status: EXACT MATCH")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    results = []

    for case in TEST_CASES:
        result = evaluate_case(case)
        if result is not None:
            results.append(result)

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION")
    print("Test cases:", len(TEST_CASES))
    print("Successful runs:", len(results))
    print("Failed runs:", len(TEST_CASES) - len(results))

    if results:
        print(
            "Macro precision:",
            f'{sum(r["precision"] for r in results) / len(results):.3f}',
        )
        print(
            "Macro recall:   ",
            f'{sum(r["recall"] for r in results) / len(results):.3f}',
        )
        print(
            "Macro F1:       ",
            f'{sum(r["f1"] for r in results) / len(results):.3f}',
        )

    print("\nNo dataset files were modified.")
    print(
        "Metrics use exact normalized text matching; "
        "semantic paraphrases require manual review."
    )


if __name__ == "__main__":
    main()
