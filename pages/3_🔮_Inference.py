import streamlit as st
import pandas as pd
import os
from utils.model_handler import load_model_and_config
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Inference", layout="wide")

def get_saved_models():
    """Scans the trained_models directory for saved model files."""
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl'))]

def main():
    """Main function for the Inference page."""
    st.title("🔮 Inference")
    st.header("Step 1: Load a Trained Model")

    saved_models = get_saved_models()
    if not saved_models:
        st.warning("No trained models found. Please train and save a model on the '⚙️ Model Selection & Training' page first.")
        return

    selected_model_file = st.selectbox("Select a trained model", options=saved_models)

    if selected_model_file:
        try:
            model, config = load_model_and_config(selected_model_file)
            st.session_state['inference_model'] = model
            st.session_state['inference_config'] = config
            st.success(f"Successfully loaded model `{config['model_name']}`.")
            st.json(config)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    if 'inference_model' in st.session_state:
        st.header("Step 2: Provide Data for Prediction")
        
        inference_method = st.radio(
            "Choose prediction method",
            ("Manual Input (Single Prediction)", "Batch Upload (CSV)"),
            horizontal=True
        )

        model = st.session_state['inference_model']
        config = st.session_state['inference_config']
        features = config['inference_features']

        if inference_method == "Manual Input (Single Prediction)":
            with st.form(key='manual_input_form'):
                st.markdown("Enter the values for the features below:")
                input_data = {}
                for feature in features:
                    # Simple check for numeric vs categorical, can be improved
                    input_data[feature] = st.number_input(f"Enter value for '{feature}'", value=0.0, format="%.4f")
                
                predict_button = st.form_submit_button("Predict")

                if predict_button:
                    try:
                        input_df = pd.DataFrame([input_data])
                        # Ensure column order matches training
                        input_df = input_df[features]

                        if config['model_name'] == "K-Nearest Neighbors (KNN)":
                             # KNN needs the training data for prediction
                            if 'processed_df' in st.session_state:
                                train_df = st.session_state['processed_df']
                                X_train = train_df[features]
                                y_train = train_df[config['target_column']]
                                knn = KNeighborsClassifier(**config['params'])
                                knn.fit(X_train, y_train)
                                prediction = knn.predict(input_df)
                                proba = knn.predict_proba(input_df)
                            else:
                                st.error("Training data not found in session for KNN. Please re-upload on page 1.")
                                return
                        else:
                            prediction = model.predict(input_df)
                            proba = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None

                        st.subheader("Prediction Result")
                        st.success(f"**Predicted Class:** `{prediction[0]}`")
                        if proba is not None:
                            st.write("**Prediction Probabilities:**")
                            st.dataframe(pd.DataFrame(proba, columns=[f"Prob_Class_{i}" for i in range(proba.shape[1])]))

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

        elif inference_method == "Batch Upload (CSV)":
            batch_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])
            if batch_file:
                try:
                    batch_df = pd.read_csv(batch_file)
                    # Validate columns
                    if not all(f in batch_df.columns for f in features):
                        st.error(f"Uploaded file is missing required columns. Required: {features}")
                    else:
                        batch_input_df = batch_df[features]
                        
                        if config['model_name'] == "K-Nearest Neighbors (KNN)":
                            if 'processed_df' in st.session_state:
                                train_df = st.session_state['processed_df']
                                X_train = train_df[features]
                                y_train = train_df[config['target_column']]
                                knn = KNeighborsClassifier(**config['params'])
                                knn.fit(X_train, y_train)
                                predictions = knn.predict(batch_input_df)
                            else:
                                st.error("Training data not found in session for KNN. Please re-upload on page 1.")
                                return
                        else:
                            predictions = model.predict(batch_input_df)
                        
                        result_df = batch_df.copy()
                        result_df['prediction'] = predictions
                        
                        st.subheader("Batch Prediction Results")
                        st.dataframe(result_df)
                        
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name='batch_predictions.csv',
                            mime='text/csv',
                        )

                except Exception as e:
                    st.error(f"An error occurred during batch prediction: {e}")

if __name__ == "__main__":
    main()
