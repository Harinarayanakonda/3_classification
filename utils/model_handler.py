import pandas as pd
import numpy as np
import os
import joblib
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Ensure the directory for saving models exists
os.makedirs("trained_models", exist_ok=True)

def preprocess_features(X):
    """Encodes categorical features for model training."""
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
    return X_processed

def find_optimal_k_for_knn(X, y, metric='euclidean', max_k=40):
    """
    Finds the optimal number of neighbors (k) for KNN using the Elbow Method
    with 5-fold cross-validation.
    """
    X_processed = preprocess_features(X)
    
    cv_scores = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_processed, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        
    optimal_k = np.argmax(cv_scores) + 1
    return optimal_k, cv_scores

def find_optimal_k_with_gridsearch(X, y, metric='euclidean', max_k=40):
    """
    Finds the optimal k for KNN using GridSearchCV.
    """
    X_processed = preprocess_features(X)
    
    param_grid = {'n_neighbors': range(1, max_k + 1)}
    knn = KNeighborsClassifier(metric=metric)
    
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_processed, y)
    
    optimal_k = grid_search.best_params_['n_neighbors']
    cv_scores = grid_search.cv_results_['mean_test_score']
    
    return optimal_k, cv_scores

def get_model(model_name, params):
    """Returns a model instance based on the name and parameters."""
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    if model_name == "Random Forest":
        return RandomForestClassifier(**params)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(**params)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(**params)
    if model_name == "XGBoost":
        return xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    if model_name == "CatBoost":
        return cb.CatBoostClassifier(**params, verbose=0)
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(**params)
    if model_name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(**params)
    raise ValueError(f"Unknown model name: {model_name}")

def train_and_evaluate_model(df, features, target, model_name, params):
    """Trains the specified model and returns the model and its classification report."""
    X = df[features]
    y = df[target]

    X_processed = preprocess_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

    model = get_model(model_name, params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return model, report

def save_model(model, config, filename, save_format):
    """Saves the trained model and its configuration."""
    if not filename:
        raise ValueError("Filename cannot be empty.")

    full_filename = f"{filename}.{save_format}" if not filename.endswith(('.joblib', '.pkl')) else filename
    config_filename = f"{os.path.splitext(full_filename)[0]}.config"
    
    model_path = os.path.join("trained_models", full_filename)
    config_path = os.path.join("trained_models", config_filename)

    if save_format == 'joblib':
        joblib.dump(model, model_path)
    elif save_format == 'pickle':
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        raise ValueError("Unsupported save format.")

    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
        
    return model_path

def load_model_and_config(model_filename):
    """Loads a model and its corresponding configuration file."""
    model_path = os.path.join("trained_models", model_filename)
    config_path = os.path.join("trained_models", f"{os.path.splitext(model_filename)[0]}.config")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Model or configuration file not found.")

    if model_filename.endswith('.joblib'):
        model = joblib.load(model_path)
    elif model_filename.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Unsupported file format.")

    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        
    return model, config