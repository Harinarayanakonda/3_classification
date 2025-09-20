import os
import joblib
import pickle
import logging
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

# Scikit-learn Imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
)

# Model Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # --- NEW: Import SVC ---
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --- Basic Logging and Directory Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs("trained_models", exist_ok=True)


def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Returns an unfitted model instance based on the name and parameters."""
    if model_name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(**params)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    if model_name == "Random Forest":
        return RandomForestClassifier(**params)
    if model_name == "AdaBoost":
        if 'estimator_max_depth' in params:
            max_depth = params.pop('estimator_max_depth', 1)
            params['estimator'] = DecisionTreeClassifier(max_depth=max_depth)
        return AdaBoostClassifier(**params)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(**params)
    if model_name == "XGBoost":
        return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
    if model_name == "CatBoost":
        return cb.CatBoostClassifier(**params, verbose=0)
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(**params)
    # --- NEW: Add Support Vector Machine (SVM) ---
    if model_name == "Support Vector Machine (SVM)":
        # Ensure probability is True for our evaluation plots
        if 'probability' not in params:
            params['probability'] = True
        return SVC(**params)
    raise ValueError(f"Unknown model name: {model_name}")


def find_best_hyperparameters(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: np.ndarray, model_name: str
) -> Dict[str, Any]:
    """Performs a randomized search to find the best hyperparameters for a model."""
    logging.info(f"Starting hyperparameter search for {model_name}...")
    
    param_dist = {}
    # ... (parameter distributions for other models) ...
    if model_name == "Random Forest":
        param_dist = {
            'classifier__n_estimators': randint(50, 500), 'classifier__max_depth': randint(5, 50),
            'classifier__min_samples_split': randint(2, 20), 'classifier__min_samples_leaf': randint(1, 10)
        }
    # --- NEW: Add search space for SVM ---
    elif model_name == "Support Vector Machine (SVM)":
        param_dist = {
            'classifier__C': uniform(0.1, 100),
            'classifier__kernel': ['rbf', 'poly', 'linear'],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__degree': randint(2, 6)
        }
    # ... (rest of the parameter distributions) ...
    else:
        logging.warning(f"No hyperparameter search space defined for {model_name}. Skipping tuning.")
        return {}

    random_search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=param_dist, n_iter=25, cv=5, 
        verbose=0, random_state=42, n_jobs=-1, scoring='accuracy'
    )
    
    random_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {random_search.best_params_}")
    
    return {key.split('__')[1]: value for key, value in random_search.best_params_.items()}


def train_and_evaluate_model(
    df: pd.DataFrame, features: List[str], target: str, model_name: str, params: Dict[str, Any],
    tune_hyperparameters: bool = False, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, float], np.ndarray, pd.Series, Dict[str, Any]]:
    """Trains a model, with optional tuning, and returns all artifacts and evaluation data."""
    X = df[features]
    y_raw = df[target]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    class_labels = list(target_encoder.classes_)
    num_classes = len(class_labels)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if tune_hyperparameters:
        temp_model = get_model(model_name, {})
        temp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', temp_model)])
        best_params = find_best_hyperparameters(temp_pipeline, X_train, y_train, model_name)
        params.update(best_params)

    final_model = get_model(model_name, params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)
    
    viz_data = {}
    optimal_threshold = 0.5
    
    if num_classes == 2:
        y_scores = y_proba[:, 1]
        precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_scores)
        
        # Align arrays for DataFrame creation
        precisions_aligned = precisions[:-1]
        recalls_aligned = recalls[:-1]
        
        # Calculate metrics for each threshold
        accuracies = [accuracy_score(y_test, (y_scores >= thr).astype(int)) for thr in thresholds_pr]
        f1_scores = 2 * (recalls_aligned * precisions_aligned) / (recalls_aligned + precisions_aligned + 1e-10)
        f2_scores = fbeta_score(y_test, (y_scores[:, np.newaxis] >= thresholds_pr).astype(int), beta=2, average=None)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds_pr[optimal_idx]
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        
        viz_data = {
            'pr_curve_df': pd.DataFrame({
                'Threshold': thresholds_pr, 'Precision': precisions_aligned,
                'Recall': recalls_aligned, 'F1-Score': f1_scores,
                'F2-Score': f2_scores, 'Accuracy': accuracies
            }),
            'roc_curve_df': pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}),
            'optimal_threshold': float(optimal_threshold)
        }
        y_pred = (y_scores >= optimal_threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "F2 Score": fbeta_score(y_test, y_pred, beta=2, average='weighted'),
    }
    try:
        if num_classes > 2:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])
    except Exception as e:
        metrics["ROC-AUC"] = "N/A"
        logging.warning(f"Could not compute ROC-AUC score: {e}")

    cm = confusion_matrix(y_test, y_pred)
    
    feature_importances = pd.Series(dtype=float)
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = pipeline.named_steps['classifier'].feature_importances_
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    except AttributeError:
        logging.warning(f"Feature importances not available for {model_name}.")
    except Exception as e:
        logging.error(f"Could not extract feature importances: {e}")

    artifacts = {"pipeline": pipeline, "target_encoder": target_encoder, "class_labels": class_labels}
    
    return artifacts, metrics, cm, feature_importances, viz_data


def save_model_artifacts(artifacts: Dict[str, Any], filename: str, save_format: str = 'joblib') -> str:
    """Saves artifacts to a file in the specified format."""
    if not (filename.endswith('.joblib') or filename.endswith('.pkl')):
        filename = f"{filename}.{save_format}"
    save_path = os.path.join("trained_models", filename)
    try:
        if save_format == 'joblib':
            joblib.dump(artifacts, save_path)
        elif save_format == 'pickle':
            with open(save_path, 'wb') as f:
                pickle.dump(artifacts, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        logging.info(f"Model artifacts saved to: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Failed to save artifacts: {e}")
        raise

def load_model_artifacts(filename: str) -> Dict[str, Any]:
    """Loads artifacts from a file, detecting the format."""
    load_path = os.path.join("trained_models", filename)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found: {load_path}")
    try:
        if filename.endswith('.joblib'):
            artifacts = joblib.load(load_path)
        elif filename.endswith('.pkl'):
            with open(load_path, 'rb') as f:
                artifacts = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        logging.info(f"Model artifacts loaded from: {load_path}")
        return artifacts
    except Exception as e:
        logging.error(f"Failed to load artifacts: {e}")
        raise