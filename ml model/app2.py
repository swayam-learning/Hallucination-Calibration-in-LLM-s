# ==========================================
# 🧠 LLM RELIABILITY ANALYZER (FINAL COMPLETE - CACHE FIXED)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

warnings.filterwarnings("ignore")

st.set_page_config(page_title="LLM Reliability Analyzer", layout="wide")

# ==========================================
# 🎨 UI STYLING
# ==========================================

st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
    margin-bottom: 20px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 LLM Reliability & Hallucination Analyzer")
st.caption("Hybrid Reliability = Uncertainty + Semantic Alignment")

# ==========================================
# 📂 LOAD DATA (CACHE REMOVED)
# ==========================================

def load_data():
    df = pd.read_csv("llm_reliability_dataset_final.csv")
    df["band"] = df["band"].map({"1B": 1, "3B": 3, "7B": 7})
    return df

df = load_data()

# ==========================================
# 🧠 TRAIN MODEL (CACHE REMOVED)
# ==========================================

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

    base = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    return calibrated, features, X_test, y_test

model, features, X_test, y_test = train_model(df)

# ==========================================
# ⚡ FAST SCORING (CACHE REMOVED)
# ==========================================

def compute_scores(df, model, features):

    X = df[features]
    uncertainty = model.predict_proba(X)[:, 1]

    cos = (df["cos_margin"] + 1) / 2
    nli = (df["nli_margin"] + 1) / 2
    semantic = (cos + nli) / 2

    final = 0.5 * uncertainty + 0.5 * semantic

    df = df.copy()
    df["uncertainty"] = uncertainty
    df["semantic"] = semantic
    df["score"] = final

    return df

df = compute_scores(df, model, features)

# ==========================================
# 🧠 BEST EXPLANATION LAYER (UNCHANGED)
# ==========================================

def generate_explanation(u, s, f):

    if f < 0.2:
        summary = "🔴 This answer is likely unreliable."
    elif f < 0.5:
        summary = "🟡 This answer has mixed reliability and should be treated with caution."
    else:
        summary = "🟢 This answer appears reliable."

    if u < 0.1 and s < 0.4:
        explanation = (
            "The model showed high uncertainty while generating this response, "
            "and the answer does not align well with known correct patterns. "
            "This combination strongly indicates that the model was unsure and likely produced an unreliable answer."
        )

    elif u < 0.1 and s > 0.6:
        explanation = (
            "Although the answer aligns well with known correct responses, "
            "the model exhibited noticeable uncertainty during generation. "
            "This suggests that while the answer may be correct, the model lacked confidence when producing it."
        )

    elif u > 0.15 and s < 0.4:
        explanation = (
            "The model appeared confident while generating this response, "
            "but the answer shows weak alignment with expected correct patterns. "
            "This pattern is often associated with confidently generated hallucinations."
        )

    elif u > 0.15 and s > 0.6:
        explanation = (
            "The model generated this answer with strong confidence, "
            "and it aligns closely with known correct responses. "
            "This combination provides strong evidence that the answer is reliable."
        )

    else:
        explanation = (
            "The model shows moderate confidence and partial alignment with known answers. "
            "These mixed signals suggest that the answer may contain both correct and uncertain elements."
        )

    details = (
        f"\n\n**Signal Breakdown:**\n"
        f"- Uncertainty Score: {u:.2f}\n"
        f"- Semantic Alignment: {s:.2f}\n"
        f"- Final Confidence: {f:.2f}"
    )

    return summary + "\n\n" + explanation + details

# ==========================================
# 📊 LOCAL SIGNAL PLOT
# ==========================================

def plot_local(u, s, f):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(["Uncertainty", "Semantic", "Final"], [u, s, f])
    ax.set_ylim(0, 1)
    ax.set_title("Signal Breakdown")
    return fig

# ==========================================
# 🎛️ SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Controls")
question = st.sidebar.selectbox("Select Question", df["question"].unique())

subset = df[df["question"] == question].sort_values("score", ascending=False)

# ==========================================
# 📊 MODEL COMPARISON (UNCHANGED)
# ==========================================

st.header("📊 Model-wise Reliability Comparison")

for _, row in subset.iterrows():

    f = row["score"]
    u = row["uncertainty"]
    s = row["semantic"]

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader(f"🤖 {row['model_name']} ({row['band']}B)")

    st.markdown("**❓ Question:**")
    st.write(row["question"])

    st.markdown("**💬 Answer:**")
    st.write(row["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Confidence", f"{f:.2f}")
    col2.metric("Uncertainty", f"{u:.2f}")
    col3.metric("Semantic", f"{s:.2f}")

    st.progress(float(f))

    if f < 0.2:
        st.error("High Risk")
    elif f < 0.5:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")

    percentile = (df["score"] < f).mean() * 100
    st.caption(f"More confident than {percentile:.1f}% of answers")

    fig_local = plot_local(u, s, f)
    st.pyplot(fig_local, use_container_width=False)

    st.markdown(generate_explanation(u, s, f))

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 📈 GLOBAL ANALYSIS (UNCHANGED)
# ==========================================

st.header("📈 Global Analysis")

st.subheader("Model-wise Average Confidence")

avg_scores = df.groupby("model_name")["score"].mean().sort_values()

fig1, ax1 = plt.subplots(figsize=(5, 3))
avg_scores.plot(kind="barh", ax=ax1)
ax1.set_title("Model Comparison")
st.pyplot(fig1, use_container_width=False)

st.subheader("Confidence Distribution")

fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.hist(df["score"], bins=20)
ax2.set_title("Score Distribution")
st.pyplot(fig2, use_container_width=False)

st.subheader("Model-wise Distribution")

fig3, ax3 = plt.subplots(figsize=(5, 3))
for name in df["model_name"].unique():
    temp = df[df["model_name"] == name]
    ax3.hist(temp["score"], bins=20, alpha=0.5, label=name)

ax3.legend()
st.pyplot(fig3, use_container_width=False)

# ==========================================
# 🔬 CALIBRATION CURVE (UNCHANGED)
# ==========================================

st.subheader("Calibration Curve (Model-wise Reliability)")

fig4, ax4 = plt.subplots(figsize=(6, 4))

ax4.plot([0, 1], [0, 1], linestyle='--', label="Perfect Calibration")

for model_name in df["model_name"].unique():

    subset_model = df[df["model_name"] == model_name]

    if len(subset_model) < 20:
        continue

    probs = subset_model["score"]
    y_true = subset_model["label_hybrid"]

    true_prob, pred_prob = calibration_curve(
        y_true, probs, n_bins=10
    )

    ax4.plot(pred_prob, true_prob, marker='o', label=model_name)

ax4.set_xlabel("Predicted Confidence")
ax4.set_ylabel("Actual Accuracy")
ax4.set_title("Calibration Curve")
ax4.legend()

st.pyplot(fig4, use_container_width=False)