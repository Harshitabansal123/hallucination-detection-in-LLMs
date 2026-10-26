"""
Run from the repo root:
    python -m experiments.test_verification
"""

from src.verification.verifier import verify_claim


# Demo cases: (claim, evidence). Includes a cross-lingual one (Hindi claim, English evidence).
DEMO = [
    (
        "Penicillin was discovered by Alexander Fleming in 1928.",
        [{"text": "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital, London.",
          "source": "demo", "language": "en"}],
    ),
    (
        "Penicillin was discovered by Alexander Fleming in 1950.",
        [{"text": "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital, London.",
          "source": "demo", "language": "en"}],
    ),
    (
        "Fleming was a Scottish-born British bacteriologist.",
        [{"text": "Penicillin is an antibiotic used to treat bacterial infections.",
          "source": "demo", "language": "en"}],
    ),
    (
        "पेनिसिलिन की खोज अलेक्जेंडर फ्लेमिंग ने 1928 में की थी।",
        [{"text": "Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital, London.",
          "source": "demo", "language": "en"}],
    ),
]


def main():
    for claim, evidence in DEMO:
        res = verify_claim(claim, evidence)
        print("\nCLAIM   :", claim)
        print("LABEL   :", res["label"])
        print(f"SUPPORT : {res['support_score']:.3f}   CONTRADICT: {res['contradict_score']:.3f}")


if __name__ == "__main__":
    main()
