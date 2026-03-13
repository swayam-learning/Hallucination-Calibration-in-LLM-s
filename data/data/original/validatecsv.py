import pandas as pd

CSV_PATH = r"data\data\raw\TruthfulQA.csv"

def validate_csv():
    df = pd.read_csv(CSV_PATH)

    required_columns = ["Question", "Correct Answers"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("[OK] CSV loaded successfully")
    print(f"[OK] Total rows: {len(df)}")
    print("[OK] Columns:", list(df.columns))

if __name__ == "__main__":
    validate_csv()
