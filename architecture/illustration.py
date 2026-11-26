"""Page illustration generator input. It is independent of the background."""


def build_prompt(sentence: str) -> str:
    if not sentence.strip():
        raise ValueError("A nonempty sentence is required.")
    return (
        "Draw a small book illustration of one subject or moment suggested "
        "by the following sentence: " + sentence
    )
