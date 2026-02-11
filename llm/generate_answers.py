import yaml
from pathlib import Path
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.db_utils import get_connection

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "data" / "raw" / "TruthfulQA.csv"

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

PROMPT_TEMPLATE = (
    "Answer the following question truthfully.\n\n"
    "Question: {question}"
)

# -------------------------------------------------
# Load config
# -------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------------------------------
# Load TruthfulQA questions
# -------------------------------------------------
def load_questions(limit=20):
    df = pd.read_csv(DATA_PATH)
    questions = df["Question"].tolist()[:limit]
    return list(enumerate(questions))

# -------------------------------------------------
# Local LLM wrapper
# -------------------------------------------------
class LocalLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

    def generate(self, prompt, max_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,   # deterministic
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------------------------------------
# Main generation loop
# -------------------------------------------------
def main():
    cfg = load_config()
    questions = load_questions(limit=20)  # SAFE START

    conn = get_connection()
    cur = conn.cursor()

    llm = LocalLLM()

    for regime in cfg["generation_settings"]["regimes"]:
        k = regime["k"]
        print(f"\n=== Local generation regime k = {k} ===")

        for qid, question in questions:
            prompt = PROMPT_TEMPLATE.format(question=question)

            for gen_idx in range(k):
                answer = llm.generate(
                    prompt=prompt,
                    max_tokens=cfg["experiment"]["max_new_tokens"],
                )

                cur.execute(
                    """
                    INSERT INTO generations (
                        question_id,
                        question,
                        answer,
                        model_name,
                        model_size,
                        temperature,
                        generation_index,
                        k_generations
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        qid,
                        question,
                        answer,
                        "llama-3.2-3b-instruct",
                        "small",
                        cfg["experiment"]["temperature"],
                        gen_idx,
                        k,
                    ),
                )

        conn.commit()

    conn.close()
    print("[OK] Local LLaMA answer generation completed successfully.")

# -------------------------------------------------
if __name__ == "__main__":
    main()
