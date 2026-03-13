import pandas as pd
import json
import os

CSV_PATH = r"data\data\raw\TruthfulQA.csv"
OUT_PATH = r"data\data\processed\questions.json"

def parse_correct_answers(cell):
    """
    Correct Answers in this CSV are semicolon-separated strings.
    This function converts them into a clean list.
    """
    if pd.isna(cell):
        return []

    # Split by semicolon and clean
    answers = [ans.strip() for ans in str(cell).split(";")]
    # Remove empty strings
    answers = [ans for ans in answers if ans]

    return answers

def prepare_questions():
    df = pd.read_csv(CSV_PATH)

    records = []

    for idx, row in df.iterrows():
        correct_answers = parse_correct_answers(row["Correct Answers"])

        if not correct_answers:
            raise ValueError(f"Row {idx}: No valid correct answers found")

        records.append({
            "question_id": idx,
            "question": str(row["Question"]).strip(),
            "correct_answers": correct_answers
        })

    os.makedirs(r"data\data\processed", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[OK] Prepared {len(records)} questions")
    print(f"[OK] Saved to {OUT_PATH}")

if __name__ == "__main__":
    prepare_questions()
