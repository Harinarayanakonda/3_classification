# utils/ui_components.py

import streamlit as st
from auth_service import auth_pb2

def render_navbar():
    """
    Renders a custom sidebar with user info, role-based navigation, and a logout button.
    This should be called on every authenticated page.
    """
    # Don't render if the user is not authenticated
    if 'authenticated' not in st.session_state or not st.session_state.get("authenticated"):
        return

    # --- NEW: HIDE THE DEFAULT STREAMLIT NAVIGATION ---
    st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)
    # --------------------------------------------------

    role_enum = st.session_state.get("role")
    role_name = auth_pb2.UserRole.Name(role_enum) if role_enum is not None else "Unknown"

    # --- Use st.sidebar to create the navigation panel on the left ---
    with st.sidebar:
        # --- User Information ---
        st.title(f"Welcome, {st.session_state.get('username', 'User')}!")
        st.caption(f"**Role:** {role_name.capitalize()}")
        st.write(st.session_state.get('email', ''))
        st.markdown("---")

        # --- Role-Based Navigation Links ---
        st.header("Navigation")

        if role_name == "ADMIN":
            st.page_link("pages/2_🔑_Admin_Dashboard.py", label="Admin Dashboard", icon="🔑")
            st.page_link("pages/1_💾_Dataset_Upload_and_Preprocessing.py", label="Dataset & Preprocessing", icon="💾")
            st.page_link("pages/2_⚙️_Model_Selection_and_Training.py", label="Model Training", icon="⚙️")
            st.page_link("pages/3_🔮_Inference.py", label="Inference", icon="🔮")

        elif role_name == "USER":
            st.page_link("pages/3_🔮_Inference.py", label="Inference", icon="🔮")

        # --- Logout Button at the bottom of the sidebar ---
        st.markdown("---")
        if st.button("Logout", key="logout_button_sidebar", use_container_width=True):
            for key in ['authenticated', 'role', 'user_id', 'token', 'username', 'email']:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("login.py")


def render_model_params(model_name: str, data_shape=None) -> dict:
    """
    Renders UI elements for model hyperparameters.
    """
    config = {}
    params = {}
    
    st.subheader(f"Hyperparameters for {model_name}")

    mode = st.radio(
        "Hyperparameter Selection Mode",
        ("Manual", "Automatic"),
        key=f"{model_name}_mode",
        horizontal=True,
        help="Choose 'Manual' to set all parameters yourself, or 'Automatic' to let the app find good parameters for you."
    )
    config['mode'] = mode

    if mode == 'Automatic':
        st.info("In Automatic mode, the best hyperparameters will be found using Randomized Search with 5-fold Cross-Validation.")
        params['n_iter'] = st.slider(
            "Number of Search Iterations", 
            min_value=10, 
            max_value=100, 
            value=20, 
            step=5,
            help="How many different combinations of parameters to try."
        )
    
    else:
        if model_name == "K-Nearest Neighbors (KNN)":
            c1, c2 = st.columns(2)
            max_k = data_shape[0] - 1 if data_shape and data_shape[0] > 1 else 50
            params['n_neighbors'] = c1.slider("n_neighbors", 1, max_k, 5)
            params['weights'] = c2.selectbox("weights", ['uniform', 'distance'])
            params['metric'] = c1.selectbox("metric", ["euclidean", "manhattan", "minkowski", "cosine"])
            params['p'] = c2.slider("p (Minkowski)", 1, 5, 2)
            params['algorithm'] = c1.selectbox("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
            params['leaf_size'] = c2.slider("leaf_size", 10, 100, 30)

        elif model_name in ["Decision Tree", "Random Forest"]:
            c1, c2 = st.columns(2)
            if model_name == "Random Forest":
                params['n_estimators'] = c1.slider("n_estimators", 10, 1000, 100)
            params['criterion'] = c2.selectbox("criterion", ["gini", "entropy", "log_loss"])
            params['max_depth'] = c1.number_input("max_depth", min_value=1, value=10)
            params['min_samples_split'] = c2.slider("min_samples_split", 2, 20, 2)
            params['min_samples_leaf'] = c1.slider("min_samples_leaf", 1, 20, 1)
            params['max_features'] = c2.selectbox("max_features", ["sqrt", "log2", None])
            if model_name == "Random Forest":
                params['bootstrap'] = c1.checkbox("bootstrap", True)
                params['n_jobs'] = c2.selectbox("n_jobs", [-1, 1, 2, 4])

        elif model_name == "AdaBoost":
            c1, c2 = st.columns(2)
            params['n_estimators'] = c1.slider("n_estimators", 10, 1000, 50)
            params['learning_rate'] = c2.slider("learning_rate", 0.01, 2.0, 1.0, 0.01)
            params['algorithm'] = c1.selectbox("algorithm", ["SAMME.R", "SAMME"])
            params['estimator_max_depth'] = c2.slider("Weak Learner Max Depth", 1, 10, 1)

        elif model_name == "Gradient Boosting":
            c1, c2 = st.columns(2)
            params['n_estimators'] = c1.slider("n_estimators", 10, 1000, 100)
            params['learning_rate'] = c2.slider("learning_rate", 0.01, 0.5, 0.1)
            params['max_depth'] = c1.slider("max_depth", 1, 15, 3)
            params['subsample'] = c2.slider("subsample", 0.5, 1.0, 1.0)
            params['min_samples_split'] = c1.slider("min_samples_split", 2, 20, 2)
            params['min_samples_leaf'] = c2.slider("min_samples_leaf", 1, 20, 1)

        if model_name not in ["K-Nearest Neighbors (KNN)"]:
            params['random_state'] = st.number_input("Random State (seed)", value=42)

    config['params'] = params
    return config


def get_model_params(model_name: str, config: dict) -> dict:
    """Extracts the parameters from the UI config dict for model instantiation."""
    return config.get('params', {})