import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_trend_model_xgb(csv_file):

    # ==============================
    # 1️⃣ Load Data
    # ==============================
    # Sort by time to ensure correct order
    df = pd.read_csv(csv_file)
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # Reset index
    df = df.reset_index(drop=True)

    # Remove Date column if present
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    # ==============================
    # 2️⃣ Define Features & Target
    # ==============================
    # Volatility-adjusted target (much better)
    volatility = df["Returns"].rolling(20).std()

     # Rolling Sharpe Ratio (shifted to avoid leakage)
    rolling_mean = df["Returns"].rolling(20).mean().shift(1)
    rolling_std = df["Returns"].rolling(20).std().shift(1)
    df["Rolling_Sharpe_20"] = rolling_mean / rolling_std

    df["Target"] = (df["Returns"].shift(-1) > volatility*0.3).astype(int)
    lag_periods = [1, 2, 3]

    lag_cols = [

        "Returns",
        "EMA_20",
        "EMA_50",
        "EMA_200",

        "RSI",

        "MACD",
        "MACD_Signal",

        "ADX",

        "Momentum",

        "Volume_Ratio",

        "Breakout_Up",
        "Breakout_Down",
        "Rolling_Sharpe_20"

    ]

    for col in lag_cols:
        for lag in lag_periods:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
   
    # Trend alignment feature (VERY POWERFUL)
    df["Trend_alignment"] = (
        (df["EMA_20"] > df["EMA_50"]) &
        (df["EMA_50"] > df["EMA_200"])
    ).astype(int)

    df["Trend_alignment_lag1"] = df["Trend_alignment"].shift(1)
    df["Trend_alignment_lag2"] = df["Trend_alignment"].shift(2)
    df["RSI_change"] = df["RSI"] - df["RSI"].shift(1)

    df["RSI_change_lag1"] = df["RSI_change"].shift(1)
    df["RSI_change_lag2"] = df["RSI_change"].shift(2)
    df = df.dropna().reset_index(drop=True)
    feature_cols = []

    for col in lag_cols:
        for lag in lag_periods:
            feature_cols.append(f"{col}_lag{lag}")

    feature_cols += [
    "Trend_alignment_lag1",
    "Trend_alignment_lag2",
    "RSI_change_lag1",
    "RSI_change_lag2"
    ]
    X = df[feature_cols]

    y = df["Target"]

    # ==============================
    # 3️⃣ Train-Test Split (Time Series Safe)
    # ==============================
    split_index = int(len(df) * 0.7)  # 70% train, 30% test

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    # ==============================
    # 4️⃣ Initialize XGBoost Model
    # ==============================
    model = XGBClassifier(

    n_estimators=800,
    max_depth=4,
    reg_alpha=1.0,     # L1
    reg_lambda=2.0,     # L2
    learning_rate=0.15,

    subsample=0.8,
    colsample_bytree=0.8,

    gamma=5,
    min_child_weight=5,

    scale_pos_weight=scale_pos_weight,

    random_state=42,
    eval_metric="logloss"
    )

    # ==============================
    # 5️⃣ Train Model
    # ==============================
    model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
    )
    # ==============================
    # 6️⃣ Predictions
    # ==============================
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities > 0.51).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]

    # ==============================
    # 7️⃣ Evaluation
    # ==============================
    accuracy = accuracy_score(y_test, predictions)
    strategy_returns = df["Returns"].iloc[split_index:] * predictions

    print("Total Strategy Return:", round(strategy_returns.sum(),4))
    print("Average Return per Trade:", round(strategy_returns.mean(),4))

    print("Model Accuracy:", round(accuracy, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))
    import matplotlib.pyplot as plt

    importance = model.feature_importances_

    plt.figure(figsize=(10,6))
    plt.barh(feature_cols, importance)
    plt.title("Feature Importance")
    plt.show()

    return model, feature_cols

def predict_next_day(model, latest_df, feature_cols):

    latest_df = latest_df.copy()

    # Make sure features are built exactly same way
    latest_row = latest_df[feature_cols].iloc[-1:]

    prob = model.predict_proba(latest_row)[:, 1][0]

    signal = 1 if prob > 0.51 else 0

    print("Next Day Probability:", round(prob,4))
    print("Predicted Signal:", signal)

    return signal, prob

# Example usage
model, feature_cols = train_trend_model_xgb("Feature_engineered_data.csv")
import joblib

joblib.dump((model, feature_cols), "trend_model.pkl")
