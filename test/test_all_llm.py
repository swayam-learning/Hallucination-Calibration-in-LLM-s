import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

TEST_PROMPT = "Answer briefly: What happens if you swallow gum?"
MAX_TOKENS = 100


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_client(provider):
    if provider == "huggingface":
        return OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
    elif provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def test_model(client, model_id):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content


def main():
    cfg = load_config()
    results = []

    for tier, models in cfg["models"].items():
        for model in models:
            name = model["name"]
            provider = model["provider"]

            # ---- BACKEND-AWARE MODEL ID ----
            model_id = model["model_id"]
            if "backend" in model:
                model_id = f"{model_id}:{model['backend']}"

            print(f"\nTesting [{tier}] {name} ({provider})")
            print(f"Using model_id: {model_id}")

            try:
                client = get_client(provider)
                response = test_model(client, model_id)
                print("Success")
                results.append((tier, name, "SUCCESS", ""))
            except Exception as e:
                print("Failed:", str(e))
                results.append((tier, name, "FAIL", str(e)))

    print("\n\n====== SUMMARY ======")
    for tier, name, status, err in results:
        print(f"[{tier.upper()}] {name:25} → {status}")


if __name__ == "__main__":
    main()
