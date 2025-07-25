import streamlit as st
import os
import pandas as pd
from utils.model_handler import (
    train_and_evaluate_model, 
    save_model, 
    find_optimal_k_for_knn,
    find_optimal_k_with_gridsearch
)
from utils.ui_components import render_model_params, get_model_params

st.set_page_config(page_title="Model Training", layout="wide")

def main():
    """Main function for the Model Selection & Training page."""
    st.title("⚙️ Model Selection & Training")

    if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
        st.warning("Please upload and process a dataset on the '💾 Dataset Upload & Preprocessing' page first.")
        return

    df = st.session_state['processed_df']
    inference_features = st.session_state['inference_features']
    target_column = st.session_state['target_column']
    
    st.info(f"Dataset loaded with **{df.shape[0]}** rows. Using **'{target_column}'** as the target and **{len(inference_features)}** features for training.")

    st.header("Step 1: Choose a Classification Algorithm")

    model_options = [
        "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", "AdaBoost",
        "Gradient Boosting", "XGBoost", "CatBoost", "LightGBM"
    ]
    selected_model = st.selectbox("Select a model", model_options)

    st.header("Step 2: Configure Model Hyperparameters")
    
    with st.expander("Click to configure hyperparameters", expanded=True):
        model_config = render_model_params(selected_model, data_shape=df.shape)

    st.header("Step 3: Train the Model")

    if st.button("Train Model", key="train_button"):
        # Special handling for KNN with automatic tuning
        if selected_model == "K-Nearest Neighbors (KNN)" and model_config.get('mode') == 'Automatic':
            auto_method = model_config.get('auto_method', 'Elbow Method')
            spinner_text = f"Finding optimal k using {auto_method}... This may take a while."
            
            with st.spinner(spinner_text):
                try:
                    X = df[inference_features]
                    y = df[target_column]
                    max_k = model_config['params']['max_k_to_check']
                    metric = model_config['params']['metric']
                    
                    if auto_method == 'Elbow Method':
                        optimal_k, cv_scores = find_optimal_k_for_knn(X, y, metric=metric, max_k=max_k)
                    else: # Grid Search CV
                        optimal_k, cv_scores = find_optimal_k_with_gridsearch(X, y, metric=metric, max_k=max_k)

                    st.success(f"Optimal number of neighbors (k) found via {auto_method}: **{optimal_k}**")
                    
                    st.subheader(f"Tuning Results ({auto_method})")
                    st.markdown("The plot shows the cross-validation accuracy for each value of k.")
                    
                    chart_data = pd.DataFrame({
                        'k': range(1, len(cv_scores) + 1),
                        'Cross-Validation Accuracy': cv_scores
                    }).set_index('k')
                    st.line_chart(chart_data)

                    final_params = {'n_neighbors': optimal_k, 'metric': metric}
                    st.session_state['trained_model'] = 'KNN_Placeholder'
                    st.session_state['model_config'] = {
                        "model_name": selected_model,
                        "params": final_params,
                        "inference_features": inference_features,
                        "target_column": target_column
                    }
                    st.info("KNN configuration with optimal k saved. You can now proceed to save the model configuration.")

                except Exception as e:
                    st.error(f"An error occurred during automatic k-selection: {e}")
        else:
            # Standard training for all other models and manual KNN
            with st.spinner("Training in progress... This may take a moment."):
                try:
                    params = get_model_params(selected_model, model_config)
                    
                    if selected_model == "K-Nearest Neighbors (KNN)":
                        st.session_state['trained_model'] = 'KNN_Placeholder'
                        st.session_state['model_config'] = {
                            "model_name": selected_model,
                            "params": params,
                            "inference_features": inference_features,
                            "target_column": target_column
                        }
                        st.success("KNN configuration saved. You can now proceed to Inference.")
                    else:
                        model, report = train_and_evaluate_model(df, inference_features, target_column, selected_model, params)
                        st.session_state['trained_model'] = model
                        st.session_state['model_config'] = {
                            "model_name": selected_model,
                            "params": params,
                            "inference_features": inference_features,
                            "target_column": target_column
                        }
                        st.success("Model trained successfully!")
                        st.subheader("Classification Report")
                        st.text(report)

                except Exception as e:
                    st.error(f"An error occurred during training: {e}")

    if st.session_state.get('trained_model') is not None:
        st.header("Step 4: Save the Model Configuration")
        st.markdown("Save your trained model configuration for future use on the Inference page.")

        with st.form(key='save_model_form'):
            model_filename = st.text_input("Enter a filename for the model", value=f"{selected_model.replace(' ', '_')}_model")
            save_format = st.radio("Select save format", options=['joblib', 'pickle'], horizontal=True)
            
            save_button = st.form_submit_button("Save Model")

            if save_button:
                try:
                    save_path = save_model(
                        st.session_state['trained_model'],
                        st.session_state['model_config'],
                        model_filename,
                        save_format
                    )
                    st.success(f"Model and configuration saved successfully to `{save_path}`")
                    st.info("You can now go to the **'🔮 Inference'** page to make predictions.", icon="➡️")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()
