import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2b-it"

class LocalLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

    def generate(self, prompt, max_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # deterministic (important for calibration)
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
