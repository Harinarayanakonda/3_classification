import streamlit as st

st.set_page_config(
    page_title="Production-Grade Classification System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'processed_df' not in st.session_state:
        st.session_state['processed_df'] = None
    if 'inference_features' not in st.session_state:
        st.session_state['inference_features'] = []
    if 'target_column' not in st.session_state:
        st.session_state['target_column'] = None
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'model_config' not in st.session_state:
        st.session_state['model_config'] = {}

def main():
    """Main function to render the welcome page."""
    initialize_session_state()

    st.title("🚀 Welcome to the Production-Grade Classification System!")
    
    st.markdown("""
    This interactive application provides a complete, end-to-end workflow for solving classification problems. 
    You can seamlessly move from uploading your data to making real-time predictions.

    ### **Workflow:**

    1.  **💾 Dataset Upload & Preprocessing:**
        - Navigate to this page using the sidebar.
        - Upload your dataset in CSV, Excel, JSON, or TXT format.
        - Preview the data and select your target variable and inference features.

    2.  **⚙️ Model Selection & Training:**
        - Choose from a wide range of powerful classification algorithms.
        - Configure model-specific hyperparameters with helpful guidance.
        - Train the model and save it for future use.

    3.  **🔮 Inference:**
        - Load your trained model.
        - Perform predictions on new data, either one by one or in a batch.

    **To get started, please select a page from the sidebar on the left.**
    """)

    st.info("This system is designed to be modular and robust, ensuring a smooth user experience. All your selections and processed data are saved across pages during your session.", icon="ℹ️")

if __name__ == "__main__":
    main()