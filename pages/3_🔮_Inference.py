# pages/3_🔮_Inference.py

import streamlit as st
import pandas as pd
import os

from utils.ui_components import render_navbar
from auth_service import auth_pb2

# --- SECURITY & NAVIGATION ---
# 1. Check if user is authenticated (accessible to both roles)
if not st.session_state.get("authenticated", False):
    st.error("🚫 You must log in first!")
    st.stop()

# 2. Render the navigation bar (it will adapt based on the user's role)
render_navbar()
# ---------------------------------

# --- YOUR ORIGINAL PAGE CODE STARTS HERE ---
st.set_page_config(page_title="Inference", layout="wide", page_icon="🔮")
st.title("🔮 Model Inference")

if 'artifacts' not in st.session_state or st.session_state.get('artifacts') is None:
    st.warning("⚠️ No model has been loaded by an admin yet.")
    st.info("An administrator must first load a model from the 'Dashboard' before inference can be performed.", icon="ℹ️")
    st.stop()

st.markdown("Select a prediction method and provide input data to get a prediction.")

active_model_name = st.session_state['artifacts']['pipeline'].steps[-1][1].__class__.__name__
st.success(f"**Model Ready:** Predictions are being made with the `{active_model_name}` model.", icon="✅")
st.markdown("---")

artifacts = st.session_state['artifacts']
pipeline = artifacts['pipeline']
target_encoder = artifacts['target_encoder']
class_labels = artifacts['class_labels']
preprocessor = pipeline.named_steps['preprocessor']
feature_names_in = preprocessor.feature_names_in_

st.header("Step 1: Choose Prediction Method")
inference_method = st.radio(
    "How would you like to provide data for prediction?",
    ("Manual Input (Single Prediction)", "Batch Upload (File)"),
    horizontal=True,
    label_visibility="collapsed"
)

if inference_method == "Manual Input (Single Prediction)":
    st.header("Step 2: Provide Input Data Manually")
    
    with st.form("manual_input_form"):
        input_data = {}
        cols = st.columns(3)
        
        if 'processed_df' in st.session_state:
            original_dtypes = {name: str(dtype) for name, dtype in st.session_state['processed_df'][feature_names_in].dtypes.items()}
        else:
            original_dtypes = {}

        for i, feature in enumerate(feature_names_in):
            col = cols[i % 3]
            dtype_str = original_dtypes.get(feature, '')
            
            if 'int' in dtype_str or 'float' in dtype_str:
                input_data[feature] = col.number_input(f"Enter {feature}", value=0.0, format="%.4f")
            else:
                input_data[feature] = col.text_input(f"Enter value for '{feature}'")

        submitted = st.form_submit_button("📈 Predict", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([input_data])[feature_names_in]
            with st.spinner("Making prediction..."):
                pred_encoded = pipeline.predict(input_df)
                pred_proba = pipeline.predict_proba(input_df)
                pred_label = target_encoder.inverse_transform(pred_encoded)[0]

                st.subheader("Prediction Result")
                st.success(f"**Predicted Class:** `{pred_label}`")
                
                st.subheader("Prediction Probabilities")
                proba_df = pd.DataFrame(pred_proba, columns=class_labels).T.rename(columns={0: 'Probability'})
                proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(proba_df)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif inference_method == "Batch Upload (File)":
    st.header("Step 2: Upload a File for Batch Prediction")
    
    batch_file = st.file_uploader(
        "Upload a CSV, Excel, JSON, or Parquet file.", 
        type=['csv', 'xlsx', 'xls', 'json', 'parquet']
    )

    if batch_file:
        try:
            file_extension = os.path.splitext(batch_file.name)[1].lower()
            if file_extension == '.csv':
                batch_df = pd.read_csv(batch_file)
            elif file_extension in ['.xlsx', '.xls']:
                batch_df = pd.read_excel(batch_file)
            elif file_extension == '.json':
                batch_df = pd.read_json(batch_file)
            elif file_extension == '.parquet':
                batch_df = pd.read_parquet(batch_file)
            
            if not all(f in batch_df.columns for f in feature_names_in):
                st.error(f"Uploaded file is missing required columns. Required: {list(feature_names_in)}")
            else:
                input_df = batch_df[feature_names_in]
                with st.spinner("Making predictions for the batch..."):
                    predictions_encoded = pipeline.predict(input_df)
                    predictions = target_encoder.inverse_transform(predictions_encoded)
                
                result_df = batch_df.copy()
                result_df['prediction'] = predictions
                
                st.subheader("Batch Prediction Results")
                st.dataframe(result_df)
                
                output_filename = f"predictions_{os.path.splitext(batch_file.name)[0]}.csv"
                csv = result_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=output_filename,
                    mime='text/csv',
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")