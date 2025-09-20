# utils/ui_components.py

import streamlit as st
import os

def render_navbar():
    """
    Renders a custom top navigation bar that adapts to the user's role.
    Automatically detects pages in the 'pages/' folder to prevent page not found errors.
    """
    # Hide default Streamlit sidebar and apply custom CSS for the navbar
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] { display: none; }
            .navbar {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 0;
                border-bottom: 1px solid #333;
            }
            .nav-links {
                display: flex;
                gap: 1.5rem; /* Spacing between nav links */
            }
        </style>
    """, unsafe_allow_html=True)

    role = st.session_state.get("role")
    
    # Use a container for navbar
    with st.container():
        col1, col2 = st.columns([8, 1])
        
        with col1:
            st.markdown('<div class="nav-links">', unsafe_allow_html=True)
            
            # Only admins get full page links
            if role == "admin":
                pages_dir = os.path.join(os.getcwd(), "pages")
                if os.path.exists(pages_dir):
                    for page_file in sorted(os.listdir(pages_dir)):
                        if page_file.endswith(".py"):
                            page_name = page_file.replace(".py", "")
                            # Replace unsafe characters for label
                            safe_label = page_name.replace("_", " ").replace("&", "and")
                            st.page_link(
                                f"pages/{page_file}",
                                label=f"**{safe_label}**",
                                icon="📄"  # generic icon
                            )
            elif role == "user":
                st.markdown("#### 🔮 Model Inference")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if st.button("Logout", key="logout_button_top"):
                st.session_state['authenticated'] = False
                st.session_state['role'] = None
                # Make sure this matches your login file
                st.switch_page("app.py")
    
    st.markdown("---")  # Visual separator


def render_model_params(model_name: str, data_shape=None) -> dict:
    """
    Renders UI elements for model hyperparameters.
    Includes 'Automatic' mode and detailed parameter explanations.
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

        # Add more models as needed: XGBoost, CatBoost, LightGBM, SVM, etc.

        if model_name not in ["K-Nearest Neighbors (KNN)"]:
            params['random_state'] = st.number_input("Random State (seed)", value=42)

    config['params'] = params
    return config


def get_model_params(model_name: str, config: dict) -> dict:
    """Extracts the parameters from the UI config dict for model instantiation."""
    return config.get('params', {})
