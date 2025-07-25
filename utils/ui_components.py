import streamlit as st

def render_model_params(model_name, data_shape=None):
    """
    Renders UI elements for model-specific hyperparameters.
    """
    config = {}
    params = {}
    
    if model_name == "K-Nearest Neighbors (KNN)":
        st.write("For KNN, you can set the number of neighbors (k) manually or find it automatically.")
        
        config['mode'] = st.radio(
            "Hyperparameter Selection Mode",
            ("Manual", "Automatic"),
            key="knn_mode",
            horizontal=True
        )

        if config['mode'] == "Automatic":
            config['auto_method'] = st.selectbox(
                "Select Automatic Method",
                ("Elbow Method", "Grid Search CV"),
                help="""
                - **Elbow Method**: A fast heuristic to find a good k value.
                - **Grid Search CV**: A more exhaustive search that is more computationally expensive but can yield better results.
                """
            )
            st.info(f"The optimal number of neighbors (k) will be determined using {config['auto_method']} with 5-fold Cross-Validation.")
            params['max_k_to_check'] = st.slider("Maximum k to check", min_value=10, max_value=100, value=40, step=5)
        
        else: # Manual mode
            max_k = data_shape[0] - 1 if data_shape and data_shape[0] > 1 else 20
            params['n_neighbors'] = st.slider("Number of Neighbors (k)", 1, max_k, 5)

        params['metric'] = st.selectbox(
            "Distance Metric",
            ["euclidean", "manhattan", "minkowski", "cosine", "hamming", "jaccard"],
            help="This metric will be used for both manual and automatic k-selection."
        )
        config['params'] = params
        return config
    
    # ... (rest of the function for other models is unchanged)
    elif model_name in ["Decision Tree", "Random Forest"]:
        params['criterion'] = st.radio("Splitting Criterion", ["gini", "entropy"], horizontal=True, help="Gini impurity is faster, while entropy can sometimes lead to more balanced trees.")
        params['max_depth'] = st.slider("Maximum Depth of Tree", 2, 50, 10)
        if model_name == "Random Forest":
            params['n_estimators'] = st.slider("Number of Trees", 10, 500, 100)

    elif model_name in ["AdaBoost", "Gradient Boosting"]:
        params['n_estimators'] = st.slider("Number of Estimators", 10, 500, 50)
        params['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        if model_name == "Gradient Boosting":
            params['loss'] = st.selectbox("Loss Function", ["log_loss", "deviance", "exponential"], help="'deviance' is equivalent to 'log_loss' for classification.")

    elif model_name == "XGBoost":
        params['objective'] = st.selectbox(
            "Objective Function",
            ["binary:logistic", "multi:softmax", "binary:hinge"],
            help="""
            - **binary:logistic**: For binary classification, outputs probability.
            - **multi:softmax**: For multi-class classification, outputs predicted class.
            - **binary:hinge**: For hinge loss (SVM-like).
            """
        )
        params['n_estimators'] = st.slider("Number of Estimators", 10, 1000, 100)
        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)

    elif model_name == "LightGBM":
        params['objective'] = st.selectbox("Objective", ["binary", "multiclass"], help="Choose based on your problem type.")
        params['n_estimators'] = st.slider("Number of Estimators", 10, 1000, 100)
        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)

    elif model_name == "CatBoost":
        params['loss_function'] = st.selectbox("Loss Function", ["Logloss", "CrossEntropy", "MultiClass"], help="Logloss for binary, CrossEntropy/MultiClass for multi-class problems.")
        params['iterations'] = st.slider("Number of Iterations", 10, 1000, 200)
        params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.05, 0.01)

    config['params'] = params
    config['mode'] = 'Manual' # Default mode for other models
    return config

def get_model_params(model_name, config):
    """Extracts the parameters from the UI config dict for model instantiation."""
    return config.get('params', {})
