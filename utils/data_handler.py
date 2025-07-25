import pandas as pd
import streamlit as st
from io import StringIO

@st.cache_data
def load_dataset(uploaded_file, manual_file_type='Auto-detect'):
    """
    Loads a dataset from an uploaded file into a pandas DataFrame.
    Uses a manually selected file type or auto-detects from the file extension.
    
    Args:
        uploaded_file: The file uploaded via st.file_uploader.
        manual_file_type (str): The file format selected by the user.
                                Can be 'Auto-detect', 'CSV', 'Excel', 'JSON', 'TXT'.
    
    Returns:
        A tuple of (pandas.DataFrame, str) representing the loaded data and its type.
    """
    if uploaded_file is None:
        return None, None

    # Determine the file type to use for parsing
    if manual_file_type == 'Auto-detect':
        file_name = uploaded_file.name
        load_type = file_name.split('.')[-1].lower()
    else:
        load_type = manual_file_type.lower()

    # Store the display name for the success message
    display_type = manual_file_type if manual_file_type != 'Auto-detect' else load_type.upper()

    try:
        if load_type == 'csv':
            return pd.read_csv(uploaded_file), display_type
        elif load_type in ['xls', 'xlsx', 'excel']:
            # Allow 'excel' from manual selection
            return pd.read_excel(uploaded_file), display_type
        elif load_type == 'json':
            return pd.read_json(uploaded_file), display_type
        elif load_type == 'txt':
            # For TXT, we assume it's CSV-like (e.g., tab or space separated).
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # Using a regex for one or more whitespace characters as a separator
            return pd.read_csv(stringio, sep=r'\s+'), display_type
        else:
            raise ValueError(f"Unsupported file type for parsing: {load_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse the file as {display_type}. Please check the file format and your selection. Error: {e}")
