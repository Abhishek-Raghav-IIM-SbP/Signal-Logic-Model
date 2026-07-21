from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import joblib
def train_regime_model(df, feature_cols):

    df["Regime_Target"] = df["Regime"].shift(-5)  # predict 5 days ahead
    df = df.dropna()
    X = df[feature_cols]
    y = df["Regime_Target"]

    split = int(len(df) * 0.7)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        eval_metric="mlogloss"
        )
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    sample_weights = y_train.map(class_weight_dict)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=50   # ← move here
        )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False
        )
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    df.loc[df.index[split:], "Predicted_Regime"] = preds

    return model, df

def validate_regimes(df):

    print("\nMean Returns by Predicted 5-Day Ahead Regime")
    print(df.groupby("Predicted_Regime")["Returns"].mean())

    print("\nVolatility by Predicted 5-Day Ahead Regime")
    print(df.groupby("Predicted_Regime")["Returns"].std())

def create_regime_labels(df):

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Regime"] = 1  # default = Sideways

    # Bull regime
    df.loc[
        (df["EMA_20"] > df["EMA_50"]) &
        (df["EMA_50"] > df["EMA_200"]),
        "Regime"
    ] = 2

    # Bear regime
    df.loc[
        (df["EMA_20"] < df["EMA_50"]) &
        (df["EMA_50"] < df["EMA_200"]),
        "Regime"
    ] = 0

    return df


# Load your dataset
df = pd.read_csv("Feature_engineered_market_data.csv")    # replace with your file name
df = create_regime_labels(df)
model, df = train_regime_model(df, ["EMA_20", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal", "ADX", "Momentum", "Volume_Ratio", "Breakout_Up", "Breakout_Down", "Rolling_Sharpe_20"])
validate_regimes(df)
df["Future_5D_Return"] = df["Returns"].rolling(5).sum().shift(-5)
feature_cols = ["EMA_20", "EMA_50", "EMA_200", "RSI", "MACD", "MACD_Signal", "ADX", "Momentum", "Volume_Ratio", "Breakout_Up", "Breakout_Down", "Rolling_Sharpe_20"]
print(df.groupby("Predicted_Regime")["Future_5D_Return"].mean())
print(df.groupby("Predicted_Regime")["Future_5D_Return"].std())
joblib.dump((model, feature_cols), "regime_model.pkl")