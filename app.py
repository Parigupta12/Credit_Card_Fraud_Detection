import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# -------------------------------
# LOAD FILES
# -------------------------------
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.pkl"

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.error("❌ Required files missing. Ensure model, scaler, features are present.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_cols = joblib.load(FEATURES_PATH)

THRESHOLD = 0.05

st.title("💳 AI Fraud Detection System")
st.write("Upload CSV / Use Sample / Create Custom Transaction")

# -------------------------------
# INPUT OPTIONS
# -------------------------------
option = st.radio(
    "Choose Input Method:",
    ["Upload CSV", "Use Sample Data", "Create Custom Transaction"]
)

data = None

# -------------------------------
# OPTION 1: UPLOAD CSV
# -------------------------------
if option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded")

# -------------------------------
# OPTION 2: SAMPLE DATA
# -------------------------------
elif option == "Use Sample Data":
    if os.path.exists("sample_data.csv"):
        data = pd.read_csv("sample_data.csv")
        st.success("✅ Sample data loaded")
    else:
        st.error("❌ sample_data.csv not found")
        st.stop()

# -------------------------------
# OPTION 3: CUSTOM INPUT
# -------------------------------
elif option == "Create Custom Transaction":

    st.subheader("✍️ Enter Transaction Details")

    values = {}

    for col in expected_cols:
        val = st.number_input(col, value=0.0)
        values[col] = val

    if st.button("➕ Add Transaction"):
        data = pd.DataFrame([values])
        st.success("✅ Transaction created")

# -------------------------------
# NO DATA
# -------------------------------
if data is None:
    st.info("📌 Please provide input data")
    st.stop()

# -------------------------------
# PREPROCESSING (MATCH NOTEBOOK)
# -------------------------------
try:
    data = data.copy()

    # Remove target if exists
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    # Convert to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill NaN
    data = data.fillna(0)

    # Add missing columns
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    # Keep correct order
    data = data[expected_cols]

    # Scale Amount (IMPORTANT)
    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

except Exception as e:
    st.error(f"❌ Preprocessing Error: {e}")
    st.stop()

# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader("📊 Data Preview")
st.write(data.head())

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Fraud"):

    try:
        probabilities = model.predict_proba(data)[:, 1]
        predictions = (probabilities >= THRESHOLD).astype(int)

        result_df = data.copy()
        result_df["Fraud_Prediction"] = predictions
        result_df["Fraud_Probability"] = probabilities

        # -------------------------------
        # RESULTS
        # -------------------------------
        st.subheader("🔍 Results")
        st.write(result_df)

        fraud_count = (predictions == 1).sum()
        normal_count = (predictions == 0).sum()

        col1, col2 = st.columns(2)
        col1.metric("✅ Normal", normal_count)
        col2.metric("🚨 Fraud", fraud_count)

        # -------------------------------
        # DOWNLOAD
        # -------------------------------
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")