# ==========================================
# 🧠 LLM RELIABILITY ANALYZER (FINAL UI)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

st.set_page_config(page_title="LLM Reliability Analyzer", layout="wide")

st.title("🧠 LLM Reliability & Hallucination Analyzer")
st.markdown("Compare multiple LLM answers and evaluate their reliability using calibrated uncertainty + semantic signals.")

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("llm_reliability_dataset_final.csv")
    df["band"] = df["band"].map({"1B": 1, "3B": 3, "7B": 7})
    return df

df = load_data()

# ==========================================
# TRAIN MODEL
# ==========================================

@st.cache_resource
def train_model(df):

    features = [
        "mean_entropy","max_entropy","entropy_variance",
        "mean_logprob","min_logprob","logprob_variance",
        "answer_length_tokens","band"
    ]

    X = df[features]
    y = df["label_hybrid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    model.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    return calibrated, features

model, features = train_model(df)

# ==========================================
# SCORING FUNCTIONS
# ==========================================

def compute_semantic_score(row):
    cos = (row["cos_margin"] + 1) / 2
    nli = (row["nli_margin"] + 1) / 2
    return np.clip((cos + nli) / 2, 0, 1)

def predict(row):
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    uncertainty = model.predict_proba(X)[0][1]
    semantic = compute_semantic_score(row)
    final = 0.5 * uncertainty + 0.5 * semantic
    return final, uncertainty, semantic

# ==========================================
# BETTER EXPLANATION LAYER
# ==========================================

def generate_explanation(row, u, s, f):

    reasons = []

    # Uncertainty explanation
    if u < 0.08:
        reasons.append("The model showed high uncertainty during generation")
    elif u < 0.15:
        reasons.append("The model had moderate uncertainty")
    else:
        reasons.append("The model was relatively confident")

    # Semantic explanation
    if s < 0.3:
        reasons.append("The answer has weak alignment with known correct responses")
    elif s < 0.6:
        reasons.append("The answer partially aligns with expected patterns")
    else:
        reasons.append("The answer strongly matches known correct answers")

    # Summary
    if f < 0.2:
        summary = "🔴 This answer is likely unreliable."
    elif f < 0.5:
        summary = "🟡 This answer has mixed reliability and should be treated with caution."
    else:
        summary = "🟢 This answer appears reliable."

    return summary + "\n\n**Reasoning:**\n- " + "\n- ".join(reasons)

# ==========================================
# RISK BAR VISUAL
# ==========================================

def show_risk_bar(score):
    st.progress(score)

# ==========================================
# UI: QUESTION SELECTOR
# ==========================================

st.sidebar.header("⚙️ Controls")

question_list = df["question"].unique()
selected_q = st.sidebar.selectbox("Select Question", question_list)

subset = df[df["question"] == selected_q].copy()

# Compute scores
subset["final_score"] = subset.apply(lambda r: predict(r)[0], axis=1)
subset = subset.sort_values("final_score", ascending=False)

# ==========================================
# MAIN DISPLAY
# ==========================================

st.header("📊 Model-wise Reliability Comparison")

for _, row in subset.iterrows():

    final, u, s = predict(row)

    with st.container():

        st.subheader(f"🤖 {row['model_name']} ({row['band']}B)")

        st.write("**Answer:**")
        st.write(row["answer"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Confidence", f"{final:.2f}")
        col2.metric("Uncertainty", f"{u:.2f}")
        col3.metric("Semantic", f"{s:.2f}")

        show_risk_bar(final)

        st.info(generate_explanation(row, u, s, final))

        st.divider()

# ==========================================
# VISUALIZATION SECTION (FIXED)
# ==========================================

st.header("📈 Global Analysis")

# Compute scores once
df["score"] = df.apply(lambda r: predict(r)[0], axis=1)

# --- Plot 1: Model Comparison ---
st.subheader("Model-wise Average Confidence")

avg_scores = df.groupby("model_name")["score"].mean().sort_values()

fig1, ax1 = plt.subplots()
avg_scores.plot(kind="barh", ax=ax1)

ax1.set_xlabel("Average Confidence Score")
ax1.set_ylabel("Model")
ax1.set_title("Model Reliability Comparison")

st.pyplot(fig1)


# --- Plot 2: Confidence Distribution ---
st.subheader("Confidence Score Distribution")

fig2, ax2 = plt.subplots()

ax2.hist(df["score"], bins=20)

ax2.set_xlabel("Confidence Score")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Reliability Scores")

st.pyplot(fig2)


# --- Plot 3: Model-wise Distribution ---
st.subheader("Confidence Distribution per Model")

fig3, ax3 = plt.subplots()

for name in df["model_name"].unique():
    subset_model = df[df["model_name"] == name]
    ax3.hist(subset_model["score"], bins=20, alpha=0.5, label=name)

ax3.legend()
ax3.set_xlabel("Confidence Score")
ax3.set_ylabel("Frequency")
ax3.set_title("Model-wise Confidence Distribution")

st.pyplot(fig3)