import streamlit as st
import polars as pl
import joblib
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import os

# Custom feature engineering transformer
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # No fitting needed for this simple transformer
        return self

    def transform(self, X):
        # Make a copy to avoid modifying original
        X = X.copy()

        # One-hot encode 'Stage_fear'
        if 'Stage_fear' in X.columns:
            dummies = pd.get_dummies(X['Stage_fear'], prefix='Stage_fear')
            X = pd.concat([X.drop(columns=['Stage_fear']), dummies], axis=1)

        # One-hot encode 'Drained_after_socializing'
        if 'Drained_after_socializing' in X.columns:
            dummies = pd.get_dummies(X['Drained_after_socializing'], prefix='Drained_after_socializing')
            X = pd.concat([X.drop(columns=['Drained_after_socializing']), dummies], axis=1)

        # Make sure all expected columns exist (add missing columns filled with 0)
        expected_cols = [
            'Time_spent_Alone',
            'Social_event_attendance',
            'Going_outside',
            'Friends_circle_size',
            'Post_frequency',
            'Stage_fear_No',
            'Stage_fear_Yes',
            'Drained_after_socializing_No',
            'Drained_after_socializing_Yes'
        ]
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0

        # Reorder columns exactly as expected by model
        X = X[expected_cols]

        return X

# Load pipeline with model and feature engineering
@st.cache_resource
def load_pipeline():
    # Load your full sklearn pipeline (with FeatureEngineering + XGB model)
    return joblib.load("../models/personality_prediction_model.pkl")

pipeline = load_pipeline()

# Prediction function
def predict(df):
    # Convert polars DataFrame to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Map "Yes"/"No" to strings for transformer to handle one-hot encoding
    for col in ['Stage_fear', 'Drained_after_socializing']:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].map({1: "Yes", 0: "No"})

    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:, 1]
    labels = ['Extrovert' if p == 1 else 'Introvert' for p in preds]
    return labels, probs

# Radar chart plot
def plot_radar(row, label):
    categories = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                  'Friends_circle_size', 'Post_frequency']
    values = row[categories].values.flatten().tolist()
    values += values[:1]  # Close the loop

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(f"Personality Radar - {label}", fontsize=14)
    ax.set_ylim(0, max(values) + 1)
    return fig

# PDF report generator
def generate_pdf(name, label, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Personality Prediction Summary", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(100, 10, f"Name: {name}")
    pdf.ln(10)
    pdf.cell(100, 10, f"Prediction: {label}")
    pdf.ln(10)
    pdf.cell(100, 10, f"Confidence: {confidence*100:.2f}%")
    pdf.ln(10)

    tmp_path = os.path.join(tempfile.gettempdir(), f"{name}_summary.pdf")
    pdf.output(tmp_path)
    return tmp_path

# Streamlit UI
st.title("🧠 HR Personality Predictor")

option = st.radio("Choose Input Method", ["📄 CSV Upload", "📝 Manual Entry"])

if option == "📄 CSV Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pl.read_csv(file)
        labels, probs = predict(df)

        result_df = df.with_columns([
            pl.Series("Prediction", labels),
            pl.Series("Confidence", [f"{p*100:.2f}%" for p in probs])
        ])

        st.dataframe(result_df)

        csv_bytes = result_df.write_csv().encode("utf-8")
        st.download_button("⬇ Download Predictions (CSV)", data=csv_bytes, file_name="predictions.csv")

        for i in range(min(3, len(result_df))):  # Show radar charts for first 3 samples
            st.subheader(f"Radar for Sample {i+1}")
            fig = plot_radar(df[i].to_pandas(), labels[i])
            st.pyplot(fig)

            pdf_path = generate_pdf(f"Sample_{i+1}", labels[i], probs[i])
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label=f"⬇ Download Summary PDF for Sample {i+1}",
                    data=f,
                    file_name=f"summary_{i+1}.pdf"
                )

else:
    st.subheader("Manual Entry Form")

    manual_input = {
        "Time_spent_Alone": st.slider("Time spent alone (hours)", 0, 12, 6),
        "Social_event_attendance": st.slider("Events attended/week", 0, 10, 3),
        "Going_outside": st.slider("Days outside/week", 0, 7, 3),
        "Friends_circle_size": st.slider("Friends in circle", 0, 15, 5),
        "Post_frequency": st.slider("Posts per week", 0, 10, 2),
        "Stage_fear": st.selectbox("Do you have stage fear?", ["Yes", "No"]),
        "Drained_after_socializing": st.selectbox("Feel drained after socializing?", ["Yes", "No"]),
    }

    if st.button("Predict"):
        input_df = pl.DataFrame([manual_input])
        label, prob = predict(input_df)
        st.success(f"🎯 Prediction: {label[0]} ({prob[0]*100:.2f}% confidence)")

        fig = plot_radar(input_df.to_pandas(), label[0])
        st.pyplot(fig)

        pdf_path = generate_pdf("Manual_User", label[0], prob[0])
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇ Download Personality PDF",
                data=f,
                file_name="summary_manual.pdf"
            )
