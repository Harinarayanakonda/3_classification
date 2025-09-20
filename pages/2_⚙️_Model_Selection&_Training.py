import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- AUTHENTICATION & AUTHORIZATION CHECK ---
# This block ensures that only logged-in administrators can access this page.
if not st.session_state.get("authenticated", False):
    st.error("Please log in first to access this page.")
    st.stop()

if st.session_state.get("role") != "admin":
    st.error("🚫 You do not have permission to view this page.")
    st.info("This page is for administrators only.")
    st.stop()
# ---------------------------------------------

# --- UTILITY IMPORTS ---
# Make sure these files exist in your utils/ directory
try:
    from utils.ui_components import render_model_params, get_model_params
    from utils.model_handler import train_and_evaluate_model, save_model_artifacts
except ImportError as e:
    st.error(f"Failed to import utility functions: {e}. Make sure your 'utils' folder is correctly set up.")
    st.stop()


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Model Training", layout="wide")


# --- PLOTTING & HELPER FUNCTIONS ---

def plot_threshold_optimization(y_true, y_scores):
    """Plots precision, recall, F1, and accuracy across a range of thresholds."""
    thresholds = np.linspace(0.01, 1.0, 100)
    metrics_list = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        precisions = precision_score(y_true, y_pred, zero_division=0)
        recalls = recall_score(y_true, y_pred, zero_division=0)
        f1s = f1_score(y_true, y_pred, zero_division=0)
        accuracies = accuracy_score(y_true, y_pred)
        metrics_list.append([t, precisions, recalls, f1s, accuracies])

    metrics_df = pd.DataFrame(metrics_list, columns=["Threshold", "Precision", "Recall", "F1-Score", "Accuracy"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_df["Threshold"], metrics_df["Precision"], label="Precision")
    ax.plot(metrics_df["Threshold"], metrics_df["Recall"], label="Recall")
    ax.plot(metrics_df["Threshold"], metrics_df["F1-Score"], label="F1-Score")
    ax.plot(metrics_df["Threshold"], metrics_df["Accuracy"], label="Accuracy")
    
    ax.set_title("Metrics vs. Threshold", fontsize=16)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_ylim(0, 1.05)

    st.pyplot(fig)


def display_metrics_guide():
    """Displays an expander with explanations for each evaluation metric."""
    with st.expander("📘 Guide to Understanding Evaluation Metrics"):
        st.markdown("""
        - **Accuracy**: The percentage of total predictions that were correct.
        - **Precision**: Of all positive predictions, how many were actually correct? (Measures false positives).
        - **Recall**: Of all actual positives, how many did the model find? (Measures false negatives).
        - **F1 Score**: A balanced score between Precision and Recall. Great for imbalanced datasets.
        - **ROC-AUC Score**: How well the model can distinguish between classes. 1.0 is perfect, 0.5 is random guessing.
        - **Confusion Matrix**: A detailed breakdown of correct and incorrect predictions for each class.
        """)


# --- MAIN APPLICATION LOGIC ---

def main():
    """Main function for the Model Selection & Training page."""
    st.title("⚙️ Model Selection & Training")

    if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
        st.warning("Please upload and process a dataset on the '💾 Dataset Upload & Preprocessing' page first.")
        return

    df = st.session_state['processed_df']
    inference_features = st.session_state['inference_features']
    target_column = st.session_state['target_column']
    
    st.info(f"Dataset loaded with **{df.shape[0]}** rows. Using **'{target_column}'** as the target.")

    # --- Step 1: Model Selection ---
    st.header("Step 1: Choose a Classification Algorithm")
    model_options = [
        "Random Forest", "XGBoost", "LightGBM", "CatBoost", 
        "Decision Tree", "Gradient Boosting", "K-Nearest Neighbors (KNN)", "AdaBoost",
        "Support Vector Machine (SVM)"
    ]
    selected_model = st.selectbox("Select a model", model_options)

    # --- Step 2: Hyperparameter Configuration ---
    st.header("Step 2: Configure Model Hyperparameters")
    with st.expander("Click to configure parameters", expanded=True):
        model_config = render_model_params(selected_model, data_shape=df.shape)
        params = get_model_params(selected_model, model_config)

    # --- Step 3: Training ---
    st.header("Step 3: Train the Model")
    if st.button("🚀 Train Model", key="train_button"):
        
        should_tune = (model_config.get('mode') == 'Automatic')
        spinner_text = "Searching for the best hyperparameters..." if should_tune else "Training model..."
        
        with st.spinner(spinner_text):
            try:
                artifacts, metrics, cm, feature_importances, viz_data = train_and_evaluate_model(
                    df=df,
                    features=inference_features,
                    target=target_column,
                    model_name=selected_model,
                    params=params,
                    tune_hyperparameters=should_tune
                )
                
                st.session_state['artifacts'] = artifacts
                st.session_state['metrics'] = metrics
                st.session_state['confusion_matrix'] = cm
                st.session_state['feature_importances'] = feature_importances
                st.session_state['viz_data'] = viz_data 
                
                st.success("Model training complete!")

            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                st.exception(e)

    # --- Step 4: Display Results ---
    if 'metrics' in st.session_state:
        st.header("Step 4: Evaluate Model Performance")

        is_binary = st.session_state['artifacts']['target_encoder'].classes_.shape[0] == 2
        if is_binary:
            viz_data = st.session_state.get('viz_data', {})
            if viz_data:
                st.subheader("📊 Performance Across Thresholds (Binary Classification)")
                
                col1, col2 = st.columns(2)

                with col1:
                    y_true = viz_data.get("y_true")
                    y_scores = viz_data.get("y_scores")
                    
                    if y_true is not None and y_scores is not None:
                        st.markdown("This plot shows how metrics change as you adjust the prediction threshold.")
                        plot_threshold_optimization(y_true, y_scores)

                with col2:
                    st.markdown("The ROC curve shows the trade-off between sensitivity and specificity.")
                    roc_curve_df = viz_data.get('roc_curve_df')
                    if roc_curve_df is not None:
                        auc_score = st.session_state['metrics'].get("ROC-AUC", 0.0)
                        fig_roc = px.area(
                            roc_curve_df, 
                            x='False Positive Rate', 
                            y='True Positive Rate',
                            title=f'ROC Curve (AUC = {auc_score:.3f})'
                        )
                        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ℹ️ Threshold Optimization and ROC Curve plots are only available for binary classification problems.")

        st.subheader("📝 Final Evaluation Metrics & Confusion Matrix")
        note = ("*(Calculated at the default 0.5 threshold for multi-class)*" 
                if not is_binary else
                f"*(Calculated at the optimal threshold of {st.session_state.get('viz_data', {}).get('optimal_threshold', 0.5):.2f})*")
        st.markdown(note)

        colA, colB = st.columns([1, 2])
        with colA:
            metrics_df = pd.DataFrame(st.session_state['metrics'].items(), columns=['Metric', 'Score'])
            st.dataframe(metrics_df, use_container_width=True)
            display_metrics_guide()
        with colB:
            fig_cm, ax_cm = plt.subplots()
            class_labels = st.session_state['artifacts']['class_labels']
            sns.heatmap(st.session_state['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Label")
            st.pyplot(fig_cm)

    if 'feature_importances' in st.session_state and not st.session_state['feature_importances'].empty:
        st.header("🧠 Feature Importance")
        fig_imp = px.bar(
            st.session_state['feature_importances'].head(15).sort_values(ascending=True),
            orientation='h', title='Top 15 Most Important Features'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    if 'artifacts' in st.session_state:
        st.header("Step 5: Save the Model")
        with st.form(key='save_model_form'):
            model_name = st.session_state['artifacts']['pipeline'].steps[-1][1].__class__.__name__
            filename = st.text_input("Enter a filename", value=f"{model_name}_model")
            save_format = st.radio("Select format", options=['joblib', 'pickle'], horizontal=True)
            
            if st.form_submit_button("💾 Save Model Artifacts"):
                try:
                    save_path = save_model_artifacts(
                        artifacts=st.session_state['artifacts'],
                        filename=filename,
                        save_format=save_format
                    )
                    st.success(f"Model artifacts saved to `{save_path}`")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")

if __name__ == "__main__":
    main()