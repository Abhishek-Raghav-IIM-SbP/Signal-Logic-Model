import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    classification_report,
)
from scipy.stats import spearmanr
import joblib
import numpy as np

BASE_FEATURE_COLS = [
    # Trend features
    "Returns",
    
    
    "ADX",
    "Rolling_Sharpe_20",
   
    # Volatility features (IMPORTANT)
    "Rolling_Vol_5",
    "Rolling_Vol_10",
    "ATR_proxy",
    "Abs_Return_lag1",
    "Vol_lag1",
    "Vol_lag2",
    "Vol_Ratio",
    "Vol_Change",
]



def _prepare_volatility_dataset(csv_file: str, feature_cols=None):
    """
    Build dataset for next‑day volatility prediction from a
    feature‑engineered CSV (e.g. Feature_engineered_market_data.csv).

    Volatility proxy = absolute next‑day return.
    """
    df = pd.read_csv(csv_file)
    df["Rolling_Vol_5"] = df["Returns"].rolling(5).std().shift(1)
    df["Rolling_Vol_10"] = df["Returns"].rolling(10).std().shift(1)
    df["ATR_proxy"] = (df["High"] - df["Low"]).shift(1)
    df["Abs_Return_lag1"] = df["Returns"].abs().shift(1)
    df["Vol_Ratio"] = df["Rolling_Vol_5"] / (df["Rolling_Vol_10"] + 1e-6)
    df["Vol_Change"] = df["Rolling_Vol_5"] - df["Rolling_Vol_10"]
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    df["Next_5D_Volatility"] = (
        df["Returns"].rolling(5).std().shift(-5)
    )
    df["Next_5D_Volatility"] = np.log(df["Next_5D_Volatility"])

    # Lagged volatility features (based on the log target)
    df["Vol_lag1"] = df["Next_5D_Volatility"].shift(1)
    df["Vol_lag2"] = df["Next_5D_Volatility"].shift(2)

    if feature_cols is None:
        feature_cols = BASE_FEATURE_COLS

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df["Next_5D_Volatility"]

    return df, X, y, feature_cols


def train_volatility_model(
    csv_file: str = "Feature_engineered_market_data.csv",
    model_path: str = "volatility_model.pkl",
):
    """
    Train an XGBoost regression model to predict 5‑day future
    realized volatility (log) using Nifty feature-engineered data.
    """
    df, X, y, feature_cols = _prepare_volatility_dataset(csv_file)

    split_idx = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Basic regression diagnostics
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"MAE (next‑day volatility): {mae:.6f}")
    print(f"R²: {r2:.4f}")

    # ==============================
    # Classification-style report on high/low volatility
    # ==============================
    # Define a threshold (median of train volatility) to create 0/1 classes
    threshold = np.quantile(y_train, 0.7)

    y_test_cls = (y_test > threshold).astype(int)
    y_pred_cls = (preds > threshold).astype(int)

    print("\nClassification Report (High vs Low Volatility):")
    print(classification_report(y_test_cls, y_pred_cls))

    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"Volatility model saved to {model_path}")

    return model, feature_cols, df


def walk_forward_evaluation(
    csv_file: str = "Feature_engineered_market_data.csv",
    model_params: dict | None = None,
    n_folds: int = 5,
    min_train: int = 400,
    test_size: int = 100,
):
    """
    Walk-forward evaluation with expanding training window.
    Returns a DataFrame of fold-level metrics.
    """
    df, X, y, feature_cols = _prepare_volatility_dataset(csv_file)

    if model_params is None:
        model_params = dict(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            objective="reg:squarederror",
        )

    results = []
    start = min_train

    while start + test_size <= len(df) and len(results) < n_folds:
        train_idx = slice(0, start)
        test_idx = slice(start, start + test_size)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        pearson = np.corrcoef(y_test, y_pred)[0, 1]
        spearman = spearmanr(y_test, y_pred).correlation

        # High-vol regime detection using rolling 70th percentile from training fold
        threshold = np.quantile(y_train, 0.7)
        y_test_cls = (y_test > threshold).astype(int)
        y_pred_cls = (y_pred > threshold).astype(int)

        cls_report = classification_report(
            y_test_cls, y_pred_cls, output_dict=True, zero_division=0
        )

        fold_result = {
            "fold_start_date": df["Date"].iloc[test_idx.start]
            if "Date" in df.columns
            else test_idx.start,
            "fold_end_date": df["Date"].iloc[test_idx.stop - 1]
            if "Date" in df.columns
            else test_idx.stop - 1,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "high_vol_recall": cls_report["1"]["recall"],
            "high_vol_precision": cls_report["1"]["precision"],
            "high_vol_f1": cls_report["1"]["f1-score"],
            "accuracy": cls_report["accuracy"],
            "macro_f1": cls_report["macro avg"]["f1-score"],
            "weighted_f1": cls_report["weighted avg"]["f1-score"],
        }

        results.append(fold_result)
        start += test_size

    results_df = pd.DataFrame(results)
    print("\nWalk-forward evaluation summary (per fold):")
    print(results_df)
    print("\nAverage across folds:")
    print(results_df.mean(numeric_only=True))

    return results_df


def load_volatility_model(model_path: str = "volatility_model.pkl"):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["feature_cols"]


def predict_next_day_volatility(model, latest_df: pd.DataFrame, feature_cols):
    """
    Predict next‑day volatility from the most recent row of
    Nifty feature‑engineered data (same columns as training CSV).
    """
    latest_df = latest_df.copy()
    latest_row = latest_df[feature_cols].iloc[-1:]
    vol_pred = float(model.predict(latest_row)[0])

    vol_pred_log = float(model.predict(latest_row)[0])
    vol_pred = np.exp(vol_pred_log)

    print(f"Predicted next-5D volatility: {vol_pred:.6f}")
    return vol_pred


if __name__ == "__main__":
    # Train on a single chronological 70/30 split and report metrics
    model, feature_cols, df_full = train_volatility_model(
        "Feature_engineered_market_data.csv"
    )

    # Walk-forward, research-style evaluation across multiple folds
    walk_forward_evaluation("Feature_engineered_market_data.csv")

    # Example: produce the latest 5-day volatility forecast
    predict_next_day_volatility(model, df_full, feature_cols)

