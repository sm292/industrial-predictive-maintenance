
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = Path("data/ai4i2020.csv")
MODEL_FILE = Path("models/failure_model.joblib")
METRICS_FILE = Path("models/metrics.json")

st.set_page_config(
    page_title="Industrial Machine Intelligence",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Industrial Machine Intelligence")
st.caption(
    "Predictive maintenance | Machine failure risk | "
    "Maintenance prioritisation"
)

if not MODEL_FILE.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_FILE)
model = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
threshold = bundle["threshold"]

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader(
    "Upload machine records (CSV)",
    type=["csv"],
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
    st.sidebar.info("Using the bundled dataset.")
else:
    st.warning("Run python download_data.py to obtain the dataset.")
    st.stop()

df.columns = df.columns.str.strip()

missing = [c for c in feature_columns if c not in df.columns]

if missing:
    st.error(
        "The uploaded CSV is missing required model inputs: "
        + ", ".join(missing)
    )
    st.stop()

# Ignore target labels and extra columns at prediction time.
X = df[feature_columns].copy()

probabilities = model.predict_proba(X)[:, 1]

results = df.copy()
results["Failure Probability"] = probabilities
results["Predicted Failure"] = (
    probabilities >= threshold
).astype(int)

def risk_category(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"

results["Risk Level"] = results["Failure Probability"].apply(
    risk_category
)

st.subheader("Operational overview")

total = len(results)
high_risk = int((results["Risk Level"] == "High").sum())
medium_risk = int((results["Risk Level"] == "Medium").sum())
mean_risk = results["Failure Probability"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records analysed", f"{total:,}")
c2.metric("High-risk records", f"{high_risk:,}")
c3.metric("Medium-risk records", f"{medium_risk:,}")
c4.metric("Mean predicted risk", f"{mean_risk:.1%}")

st.info(
    "Risk bands are initial demonstration settings, not validated "
    "maintenance thresholds. Calibrate them against operational costs "
    "and held-out model performance before real use."
)

left, right = st.columns(2)

with left:
    st.subheader("Predicted risk distribution")
    risk_counts = (
        results["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Level")
        .reset_index(name="Machine records")
    )

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Machine records",
        color="Risk Level",
        title="Records by risk category",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Failure probability")
    fig = px.histogram(
        results,
        x="Failure Probability",
        nbins=30,
        title="Predicted probability distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Maintenance priority list")

display_columns = [
    c for c in ["UDI", "Product ID", "Type"] if c in results.columns
]

display_columns += [
    "Failure Probability",
    "Risk Level",
    "Predicted Failure",
]

ranked = results.sort_values(
    "Failure Probability",
    ascending=False,
)

st.dataframe(
    ranked[display_columns].head(100),
    use_container_width=True,
)

st.subheader("Explore operating conditions")

numeric_columns = X.select_dtypes(include="number").columns.tolist()

if numeric_columns:
    selected = st.selectbox(
        "Select a machine measurement",
        numeric_columns,
    )

    fig = px.scatter(
        results,
        x=selected,
        y="Failure Probability",
        color="Risk Level",
        hover_data=[
            c for c in ["UDI", "Product ID"] if c in results.columns
        ],
        title=f"{selected} versus predicted failure probability",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Download results")

csv_data = ranked.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download machine risk report (CSV)",
    data=csv_data,
    file_name="machine_failure_risk_report.csv",
    mime="text/csv",
)

if METRICS_FILE.exists():
    import json

    with open(METRICS_FILE, encoding="utf-8") as f:
        metrics = json.load(f)

    st.subheader("Model evaluation")
    st.write(f"Selected model: **{metrics['selected_model']}**")

    metric_df = pd.DataFrame(metrics["results"])
    st.dataframe(
        metric_df[
            ["model", "precision", "recall", "f1", "pr_auc", "roc_auc"]
        ].round(3),
        use_container_width=True,
    )

st.caption(
    "Research prototype using the UCI AI4I synthetic dataset. "
    "Predictions are not validated for real industrial maintenance."
)
