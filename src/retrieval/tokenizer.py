import regex


def tokenize(text):
    return regex.findall(
        r"[\p{L}\p{M}\p{N}]+",
        text.lower()
    )