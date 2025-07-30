import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
from Feature_engineering import FeatureEngineering
import plotly.graph_objects as go

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("../models/personality_prediction_model2.pkl")

pipeline = load_model()
st.set_page_config(page_title="HR Personality Predictor", layout="centered")

# --- Title and Mode Switch ---
st.title("🧠 HR Personality Predictor")
st.markdown("Upload a CSV or manually enter details to predict if someone is an **Introvert** or **Extrovert**.")

input_mode = st.radio("Choose Input Method", ["Manual Input", "CSV Upload"])
def plot_radar_chart(features_dict: dict, personality: str):
                            categories = list(features_dict.keys())
                            values = list(features_dict.values())

                            # Repeat first element to close the loop in radar chart
                            values += values[:1]
                            categories += categories[:1]

                            fig = go.Figure(
                                data=[
                                    go.Scatterpolar(
                                        r=values,
                                        theta=categories,
                                        fill='toself',
                                        name='Traits',
                                        line=dict(color='royalblue')
                                    )
                                ]
                            )

                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 10]  # You can adjust this based on your feature scale
                                    )
                                ),
                                showlegend=False,
                                title=f"Personality: {personality}"
                            )
                            return fig

# --- Manual Input Mode ---
if input_mode == "Manual Input":
    st.subheader("📝 Enter Person Details")

    with st.form("prediction_form"):
        time_spent_alone = st.slider("Time spent alone (1–10)", 1, 10, 5)
        stage_fear = st.selectbox("Stage fear?", ["Yes", "No"])
        social_event_attendance = st.slider("Social event attendance (1–10)", 1, 10, 5)
        going_outside = st.slider("Going outside (1–10)", 1, 10, 5)
        drained_after_socializing = st.selectbox("Drained after socializing?", ["Yes", "No"])
        friends_circle_size = st.slider("Friends circle size (1–10)", 1, 10, 5)
        post_frequency = st.slider("Post frequency (1–10)", 1, 10, 5)

        submit_btn = st.form_submit_button("Predict Personality")

        if submit_btn:
            with st.spinner("Predicting..."):
                try:
                    raw_input_dict = {
                        "Time_spent_Alone": time_spent_alone,
                        "Stage_fear": stage_fear,
                        "Social_event_attendance": social_event_attendance,
                        "Going_outside": going_outside,
                        "Drained_after_socializing": drained_after_socializing,
                        "Friends_circle_size": friends_circle_size,
                        "Post_frequency": post_frequency,
                    }

                    input_df = pl.DataFrame([raw_input_dict])
                    
                    # Get feature names from model
                    xgb_model = pipeline.named_steps["xgb_classifier"]
                    expected_features = xgb_model.feature_names_in_

                    # Transform the input using the pipeline up to the feature_engineering step
                    transformed = pipeline.named_steps["feature_engineering"].transform(input_df)

                    # Ensure all expected columns exist in the transformed DataFrame
                    for col in expected_features:
                        if col not in transformed.columns:
                            transformed = transformed.with_columns(pl.lit(0).alias(col))

                    # Reorder columns to match the model input
                    transformed = transformed.select(expected_features)

                    # Predict
                    prediction = xgb_model.predict(transformed.to_pandas())[0]
                    result = "Extrovert" if prediction == 1 else "Introvert"

                    result = "Extrovert" if prediction == 1 else "Introvert"

                    st.success(f"🎯 Predicted Personality: **{result}**")
                    st.balloons()

                    # Convert binary Yes/No to numeric for plotting
                    binary_map = {"Yes": 1, "No": 0}

                    # Prepare feature values for radar chart
                    radar_features = {
                        "Time Alone": time_spent_alone,
                        "Social Events": social_event_attendance,
                        "Going Outside": going_outside,
                        "Friend Circle Size": friends_circle_size,
                        "Post Frequency": post_frequency,
                        "Drained After Socializing": binary_map.get(drained_after_socializing, 0)
                    }

                    # Show radar chart
                    st.subheader("🧭 Behavioral Radar Chart")
                    radar_fig = plot_radar_chart(radar_features, result)
                    st.plotly_chart(radar_fig, use_container_width=True)


                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.write("Debug - Input DataFrame Columns:", input_df.columns)
                    st.write("Debug - Input DataFrame Shape:", input_df.shape)
                    st.write(input_df)

# --- CSV Upload Mode ---
else:
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with the correct format", type=["csv"])

    if uploaded_file:
        df_raw = pl.read_csv(uploaded_file)

        required_cols = [
            "id", "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
            "Going_outside", "Drained_after_socializing", "Friends_circle_size", "Post_frequency"
        ]
        if not all(col in df_raw.columns for col in required_cols):
            st.error("❌ CSV must contain the following columns:\n" + ", ".join(required_cols))
        else:
            st.write("📄 Data Preview:")
            st.dataframe(df_raw.head().to_pandas())

            if st.button("🔮 Predict for All"):
                with st.spinner("Running predictions..."):
                    try:
                        ids = df_raw["id"]
                        input_df = df_raw.drop("id")

                        preds = pipeline.predict(input_df)
                        personalities = ["Extrovert" if p == 1 else "Introvert" for p in preds]

                        final = pl.DataFrame({
                            "id": ids,
                            "Personality": pl.Series(personalities)
                        })

                        st.success("✅ Prediction Complete!")
                        st.dataframe(final)
                        st.balloons()



                        # 📊 Chart
                        st.subheader("📊 Personality Distribution")
                        fig, ax = plt.subplots()
                        pd.Series(personalities).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"])
                        ax.set_ylabel("Count")
                        ax.set_title("Predicted Personality")
                        st.pyplot(fig)

                        # ⬇️ CSV Download
                        df_result = final.to_pandas()
                        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download CSV", csv_bytes, file_name="personality_predictions.csv", mime="text/csv")

                        # 📄 PDF Report Generator
                        def generate_pdf(data: pd.DataFrame):
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.cell(200, 10, "Personality Prediction Report", ln=True, align="C")
                            pdf.ln(10)
                            for i, row in data.iterrows():
                                summary = f"{i+1}. ID: {row['id']}, Personality: {row['Personality']}"
                                pdf.multi_cell(0, 10, summary)
                                pdf.ln(2)
                            tmp_path = tempfile.mktemp(suffix=".pdf")
                            pdf.output(tmp_path)
                            return tmp_path


                        pdf_path = generate_pdf(df_result)
                        with open(pdf_path, "rb") as f:
                            st.download_button("📄 Download PDF Report", f, file_name="personality_report.pdf", mime="application/pdf")

                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.write("Debug - Data Columns:", input_df.columns)
                        st.write("Debug - Data Shape:", input_df.shape)
                    
