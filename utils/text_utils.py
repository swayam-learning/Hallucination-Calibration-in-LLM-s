import re

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - remove extra spaces
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_answer(answer: str, reference_answers: list) -> bool:
    """
    Checks if generated answer contains
    any of the reference correct answers.
    """
    answer = normalize_text(answer)
    for ref in reference_answers:
        ref = normalize_text(ref)
        if ref in answer:
            return True
    return False
