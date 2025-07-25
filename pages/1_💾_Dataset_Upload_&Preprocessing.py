import streamlit as st
import pandas as pd
from utils.data_handler import load_dataset

st.set_page_config(page_title="Dataset Upload", layout="wide")

def main():
    """Main function for the Dataset Upload & Preprocessing page."""
    st.title("💾 Dataset Upload & Preprocessing")
    
    st.header("Step 1: Upload Your Dataset")
    st.markdown("Upload your data and specify its format if needed.")

    # Dropdown for manual file format selection
    file_format = st.selectbox(
        "Select file format (or leave as Auto-detect)",
        options=['Auto-detect', 'CSV', 'Excel', 'JSON', 'TXT'],
        index=0,
        help="Select 'Auto-detect' to infer the format from the file extension. Choose a specific format if the extension is incorrect or missing."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xls', 'xlsx', 'json', 'txt'],
        help="Supported formats: CSV, Excel, JSON, TXT"
    )

    if uploaded_file:
        try:
            # Pass the selected format to the load function
            df, file_type = load_dataset(uploaded_file, manual_file_type=file_format)
            st.session_state['df'] = df
            st.success(f"Successfully loaded a `{file_type}` file with {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return

    if st.session_state.get('df') is not None:
        df = st.session_state['df']
        st.header("Step 2: Preview Your Dataset")
        st.markdown("Here are the first 5 records of your uploaded dataset.")
        st.dataframe(df.head())

        st.header("Step 3: Preprocess and Select Features")
        st.markdown("Select the target variable and features to be used for inference.")
        
        all_columns = df.columns.tolist()

        # Using a form for better organization
        with st.form(key='preprocessing_form'):
            target_column = st.selectbox(
                "Select the Target Variable",
                options=all_columns,
                index=len(all_columns)-1 if all_columns else 0,
                help="This is the column the model will learn to predict."
            )

            columns_to_drop = st.multiselect(
                "Select columns to drop (if any)",
                options=[col for col in all_columns if col != target_column],
                help="Select any columns that are not needed for training or inference (e.g., IDs, irrelevant data)."
            )
            
            submit_button = st.form_submit_button(label='Confirm Selections & Continue')

        if submit_button:
            st.session_state['target_column'] = target_column
            
            # Process the dataframe
            processed_df = df.drop(columns=columns_to_drop)
            st.session_state['processed_df'] = processed_df
            
            # Define inference features
            inference_features = [col for col in processed_df.columns if col != target_column]
            st.session_state['inference_features'] = inference_features
            
            st.success("Selections saved! You can now proceed to Model Selection & Training.")
            st.subheader("Configuration Summary:")
            st.json({
                "Target Variable": target_column,
                "Dropped Columns": columns_to_drop,
                "Inference Features": inference_features,
                "Processed Data Shape": processed_df.shape
            })
            st.info("Navigate to the **'⚙️ Model Selection & Training'** page from the sidebar to continue.", icon="➡️")

if __name__ == "__main__":
    main()
