import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from indicators import add_indicators
from strategy import generate_signals
from backtester import backtest
from optimizer import parameter_search
from indicator_sets import indicator_sets

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    required_cols = ['Open','High','Low','Close','Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Data file must contain Date,Open,High,Low,Close,Volume columns.")
    return df

def visualize_data_with_indicators(df):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df.index, df['Close'], label='Close', color='black')
    axes[0].plot(df.index, df['EMA'], label='EMA', color='blue', alpha=0.7)
    axes[0].set_title('Price and EMA')
    axes[0].legend()

    axes[1].plot(df.index, df['RSI'], label='RSI', color='green')
    axes[1].axhline(70, color='red', linestyle='--')
    axes[1].axhline(30, color='green', linestyle='--')
    axes[1].set_title('RSI')

    axes[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_title('MACD')

    axes[3].plot(df.index, df['ADX'], label='ADX', color='brown')
    axes[3].axhline(20, color='grey', linestyle='--')
    axes[3].axhline(25, color='grey', linestyle='--')
    axes[3].set_title('ADX')

    plt.tight_layout()
    plt.show()

def log_results(message, log_file="indicator_test_results.log"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - {message}\n")

def main():
    config = load_config("config.json")
    data_path = os.path.join("data", "SPY_5min_preprocessed.csv")
    df = load_data(data_path)

    # Add core indicators and visualize
    df = add_indicators(df, config)
    visualize_data_with_indicators(df)

    # Generate signals and backtest
    df = generate_signals(df, config)
    results = backtest(df, config)
    print("Backtest Results:")
    print(results)

    # Parameter optimization example
    param_grid = {
        "rsi_threshold_bearish": [65, 70, 75],
        "rsi_threshold_bullish": [25, 30, 35]
    }
    best_params, best_performance = parameter_search(df, config, param_grid)
    print("Best Parameters Found:", best_params)
    print("Best Performance (Final Equity):", best_performance)

    # Now test multiple indicator sets for classification accuracy
    log_file = "indicator_test_results.log"
    with open(log_file, "w") as f:
        f.write("Indicator Test Results Log\n")

    # Create prediction target: next candle up or down
    df['Future_Close'] = df['Close'].shift(-1)
    df['Up_Indicator'] = (df['Future_Close'] > df['Close']).astype(int)
    df = df.dropna(subset=['Future_Close'])

    train_size = int(len(df)*0.7)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    for set_name, func in indicator_sets.items():
        # Apply the indicator set to train/test
        train = df_train.copy()
        test = df_test.copy()

        train = func(train)
        test = func(test)

        # Ensure columns align
        test = test.reindex(columns=train.columns)
        test = test.dropna()
        if len(test) == 0 or len(train) == 0:
            log_results(f"{set_name}: Not enough data after adding indicators.", log_file)
            continue

        base_cols = ['Open','High','Low','Close','Volume','Future_Close','Up_Indicator']
        feature_cols = [c for c in train.columns if c not in base_cols]

        X_train = train[feature_cols]
        y_train = train['Up_Indicator']
        X_test = test[feature_cols]
        y_test = test['Up_Indicator']

        # Train a simple logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        result_message = f"{set_name}: Accuracy = {acc:.4f}"
        print(result_message)
        log_results(result_message, log_file)

if __name__ == "__main__":
    main()
