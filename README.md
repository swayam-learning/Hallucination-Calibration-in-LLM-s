# Hallucination-Calibration-in-LLM-s
# Confidence-Calibrated Hallucination Detection in Large Language Models

This repository contains the code and experiments for a research project on **detecting and calibrating hallucinations in large language models (LLMs)**.  
The project focuses on **when LLM outputs should not be trusted**, rather than proposing a new benchmark or improving model accuracy.

---

## 1. Motivation

Large Language Models (LLMs) often generate fluent and confident answers that may be factually incorrect, a phenomenon commonly referred to as **hallucination**.  
In high-stakes settings (education, healthcare, research, decision support), **confident false answers are more dangerous than uncertain or abstained responses**.

Most existing work evaluates *whether* an answer is correct.  
This project instead asks:

> **How can we estimate how risky it is to trust an LLM answer without verification?**

---

## 2. Project Goal

The goal of this project is to build a **post-hoc reliability estimation and calibration framework** that:

- operates on **black-box LLM outputs**
- does **not fine-tune or modify** the underlying models
- provides **confidence signals aligned with actual correctness**
- helps users decide **when external verification is required**

This project **does not introduce a new benchmark** and **does not rank models**.

---

## 3. Key Research Ideas

- **Hallucination ≠ simple error**  
  The most dangerous failures are *confident, fluent falsehoods*.

- **Inverse scaling (TruthfulQA insight)**  
  Larger models may hallucinate more because they imitate human misconceptions more confidently.

- **Cost asymmetry**  
  False confidence is more harmful than false uncertainty.  
  Over-warning is preferable to over-trust in high-stakes scenarios.

- **Calibration over accuracy**  
  The focus is on aligning confidence with correctness, not maximizing raw performance.

---

## 4. Model Setup

Hallucination behavior is evaluated across **multiple model scales and families**, using **API-based, black-box access only**.

### Small Models (≈1–2B)
- `meta-llama/Llama-3.2-1B-Instruct` (Hugging Face API)
- `google/gemma-3n-e2b-it` (effective ~2B, free API)

### Medium Models (7B–13B)
- `mistralai/mistral-7b-instruct-v0.2`
- `google/gemma-13b-it`

### Large Models (≥30B)
- `nvidia/nemotron-30b`
- `mixtral-8x7b-instruct`

Each model is evaluated under the **same prompting and generation protocol** to ensure fair comparison.

---

## 5. Experimental Protocol

- Dataset: **TruthfulQA**
- Setting: **Zero-shot**, no fine-tuning
- Generations: **3 independent responses per question–model pair**
- Decoding:
  - primary runs use deterministic decoding (temperature = 0)
- Storage:
  - **All raw generations are preserved**
  - CSV files for inspection
  - SQLite database for structured analysis

Raw outputs are frozen before feature extraction or modeling.

---

## 6. Repository Structure

```text
hallucination-calibration/
│
├── config/                 # Configuration files
│   └── config.yaml
│
├── data/
│   ├── raw/                # Original datasets (e.g., TruthfulQA CSV)
│   ├── processed/          # Processed questions
│   └── data.db             # SQLite database
│
├── llm/                    # LLM interface layer
│   ├── base_client.py
│   ├── hf_client.py
│   ├── openrouter_client.py
│   └── generate_answers.py
│
├── features/               # Feature extraction
│   ├── entropy_features.py
│   ├── consistency_features.py
│   ├── semantic_features.py
│   └── linguistic_features.py
│
├── models/                 # Reliability prediction models
│   ├── train_reliability_model.py
│   └── evaluate_model.py
│
├── calibration/            # Confidence calibration
│   ├── temperature_scaling.py
│   └── calibration_metrics.py
│
├── utils/                  # Utilities
│   ├── db_utils.py
│   ├── text_utils.py
│   └── metrics.py
│
├── run_pipeline.py
└── requirements.txt
