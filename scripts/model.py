# model.py
# ===============================================================
#   MACHINE LEARNING MODELS FOR INSURANCE RISK & PRICING
#   Customized for your dataset structure
# ===============================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt


# ===============================================================
# 1. SMART TRAIN/TEST SPLIT
# ===============================================================

def split_data(X, y, target_type="regression", test_size=0.2, random_state=42):
    """
    - classification → stratified split
    - regression → normal split

    target_type:
      - "risk" → binary (HasClaim)
      - "severity" → continuous (ClaimSeverity)
      - "margin" → continuous (Margin)
    """
    if target_type == "risk":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in splitter.split(X, y):
            return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ===============================================================
# 2. TRAIN SUITE OF MODELS
# ===============================================================

def train_models(X_train, y_train, target_type="regression"):
    """
    Initializes and trains 4 ML models.
    Special XGBoost configuration is applied for:
      - risk models (logistic loss)
      - severity models (gamma regression)
      - pricing/margin models (squared loss)
    """

    # -----------------------------
    # BASE MODELS
    # -----------------------------
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42)

    # -----------------------------
    # XGBOOST ENGINE TUNED FOR INSURANCE
    # -----------------------------
    if target_type == "risk":
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )
    elif target_type == "severity":
        xgb_model = xgb.XGBRegressor(
            objective="reg:gamma",
            eval_metric="rmse",
            random_state=42
        )
    else:
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42
        )

    # -----------------------------
    # TRAIN ALL MODELS
    # -----------------------------
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return lr, dt, rf, xgb_model


# ===============================================================
# 3. EVALUATION FUNCTION
# ===============================================================

def evaluate_model(model, X_test, y_test, target_type="regression"):
    """
    Supports:
      - regression (severity, premium, margin)
      - classification (risk)
    """

    y_pred = model.predict(X_test)

    # ---------- Classification Metrics ----------
    if target_type == "risk":
        from sklearn.metrics import accuracy_score, roc_auc_score

        # Ensure predictions are probabilities
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred

        return {
            "accuracy": accuracy_score(y_test, (y_pred > 0.5).astype(int)),
            "ROC_AUC": roc_auc_score(y_test, y_score),
            "y_pred": y_score
        }

    # ---------- Regression Metrics ----------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "y_pred": y_pred
    }


# ===============================================================
# 4. PLOT COMPARATIVE METRICS
# ===============================================================

def plot_metrics(models, mae, mse, rmse, r2):
    """
    Visual comparison of regression models.
    """

    plt.figure(figsize=(6, 4))
    plt.bar(models, mae, color='skyblue')
    plt.title("MAE Comparison")
    plt.ylabel("MAE")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, mse, color='orange')
    plt.title("MSE Comparison")
    plt.ylabel("MSE")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, rmse, color='purple')
    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, r2, color='green')
    plt.title("R2 Score Comparison")
    plt.ylabel("R2")
    plt.show()
