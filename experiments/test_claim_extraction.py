from src.claims.pipeline import (
    process_question,
    save_result
)


def main():

    print(
        "========================================"
    )

    print(
        "   MULTILINGUAL CLAIM EXTRACTION"
    )

    print(
        "========================================"
    )


    # --------------------------------
    # User Question
    # --------------------------------

    question = input(
        "\nEnter your question: "
    )


    # --------------------------------
    # Language
    # --------------------------------

    language = input(
        "Language code [en]: "
    ).strip() or "en"


    # --------------------------------
    # Process
    # --------------------------------

    result = process_question(
        question,
        language
    )


    # --------------------------------
    # Display Answer
    # --------------------------------

    print(
        "\n========== LANGUAGE =========="
    )

    print(
        result["language"]
    )


    print(
        "\n========== LLM ANSWER =========="
    )

    print(
        result["answer"]
    )


    # --------------------------------
    # Display Claims
    # --------------------------------

    print(
        "\n========== EXTRACTED CLAIMS =========="
    )


    if not result["claims"]:

        print(
            "No factual claims found."
        )

    else:

        for claim in result["claims"]:

            print(
                f"{claim['claim_id']}. "
                f"{claim['claim']}"
            )


    # --------------------------------
    # Save
    # --------------------------------

    output_file = save_result(
        result
    )


    print(
        "\n========== SAVED =========="
    )

    print(
        f"Output saved to: {output_file}"
    )


if __name__ == "__main__":

    main()