import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tabulate import tabulate

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

import xgboost as xgb
import optuna
from optuna.integration import KerasPruningCallback

# Reinforcement Learning
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##############################
# 1. Data Loading & Indicators
##############################
def load_data(file_path):
    logging.info(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path, parse_dates=['time'])
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)

    rename_mapping = {
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    }
    data.rename(columns=rename_mapping, inplace=True)

    # Sort by Date
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    logging.info(f"Data columns after renaming: {data.columns.tolist()}")
    logging.info("Data loaded and sorted successfully.")

    return data

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram

def compute_adx(df, window=14):
    """
    Example ADX calculation (pseudo-real):
    You can implement a full ADX formula if you’d like.
    Here, we do a slightly more robust approach than rolling std.
    """
    # True range
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    tr = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    tr_rolling = tr.rolling(window=window).mean()

    # Simplistic to replicate ADX-like effect
    adx_placeholder = tr_rolling / df['Close']
    df.drop(['H-L','H-Cp','L-Cp'], axis=1, inplace=True)
    return adx_placeholder

def compute_obv(df):
    # On-Balance Volume
    signed_volume = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    return signed_volume.cumsum()

def compute_bollinger_bands(series, window=20, num_std=2):
    """
    Bollinger Bands: middle=MA, upper=MA+2*std, lower=MA-2*std
    Return the band width or separate columns.
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma  # optional metric
    return upper, lower, bandwidth

def compute_mfi(df, window=14):
    """
    Money Flow Index: uses typical price, volume, direction.
    For demonstration.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    # Positive or negative
    df_shift = typical_price.shift(1)
    flow_positive = money_flow.where(typical_price > df_shift, 0)
    flow_negative = money_flow.where(typical_price < df_shift, 0)
    # Sum over window
    pos_sum = flow_positive.rolling(window=window).sum()
    neg_sum = flow_negative.rolling(window=window).sum()
    # RSI-like formula
    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-9)))
    return mfi

def calculate_technical_indicators(df):
    logging.info("Calculating technical indicators...")

    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['MACD'] = compute_macd(df['Close'])
    df['OBV'] = compute_obv(df)
    df['ADX'] = compute_adx(df)
    
    # Bollinger
    upper_bb, lower_bb, bb_width = compute_bollinger_bands(df['Close'], window=20)
    df['BB_Upper'] = upper_bb
    df['BB_Lower'] = lower_bb
    df['BB_Width'] = bb_width

    # MFI
    df['MFI'] = compute_mfi(df)

    # Simple/EMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    # STD
    df['STDDEV_5'] = df['Close'].rolling(window=5).std()
    
    df.dropna(inplace=True)
    logging.info("Technical indicators calculated successfully.")
    return df

##############################
# 2. Parse Arguments
##############################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train LSTM and DQN models for stock trading.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV data file.')
    return parser.parse_args()

##############################
# 3. Main
##############################
def main():
    args = parse_arguments()
    csv_path = args.csv_path

    # 1) Load data
    data = load_data(csv_path)
    data = calculate_technical_indicators(data)

    # 2) Build feature set
    # We deliberately EXCLUDE 'Close' from the features so the model doesn't trivially see it.
    # Instead, rely on advanced indicators + OHLC + Volume.
    feature_columns = [
        'Open', 'High', 'Low', 'Volume',
        'RSI', 'MACD', 'OBV', 'ADX',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'MFI', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'STDDEV_5'
    ]
    target_column = 'Close'  # still used for label/evaluation

    # Keep only these columns + Date + target
    data = data[['Date'] + feature_columns + [target_column]].dropna()

    # 3) Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(data[feature_columns])
    scaled_target = scaler_target.fit_transform(data[[target_column]]).flatten()

    # 4) Create sequences for LSTM
    def create_sequences(features, target, window_size=15):
        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i+window_size])
            y.append(target[i+window_size])
        return np.array(X), np.array(y)

    window_size = 15
    X, y = create_sequences(scaled_features, scaled_target, window_size)

    # 5) Train/Val/Test Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    test_size = len(X) - train_size - val_size

    X_train, X_val, X_test = (
        X[:train_size],
        X[train_size:train_size+val_size],
        X[train_size+val_size:]
    )
    y_train, y_val, y_test = (
        y[:train_size],
        y[train_size:train_size+val_size],
        y[train_size+val_size:]
    )

    logging.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    logging.info(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    # 6) Device Config
    def configure_device():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"{len(gpus)} GPU(s) detected and configured.")
            except RuntimeError as e:
                logging.error(e)
        else:
            logging.info("No GPU detected, using CPU.")
    configure_device()

    # 7) Build LSTM
    from tensorflow.keras.regularizers import l2
    def build_lstm(input_shape, units=128, dropout=0.3, lr=1e-3):
        model = Sequential()
        # Example: 2 stacked LSTM layers
        model.add(Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l2(1e-4)), input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=l2(1e-4))))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=lr)
        model.compile(loss=Huber(), optimizer=optimizer, metrics=['mae'])
        return model

    # 8) Train LSTM (you can still do Optuna if you like, omitted here for brevity)
    model_lstm = build_lstm((X_train.shape[1], X_train.shape[2]), units=128, dropout=0.3, lr=1e-3)

    early_stop = EarlyStopping(patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

    model_lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 9) Evaluate
    def evaluate_lstm(model, X_test, y_test):
        y_pred_scaled = model.predict(X_test).flatten()
        # If we forcibly clamp predictions to [0,1], do so, else skip:
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_true = scaler_target.inverse_transform(y_test.reshape(-1,1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Direction
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        directional_acc = np.mean(direction_true == direction_pred)

        logging.info(f"LSTM Test -> MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, DirAcc={directional_acc:.4f}")

        # Quick Plot
        plt.figure(figsize=(12,6))
        plt.plot(y_true[:100], label='Actual')
        plt.plot(y_pred[:100], label='Predicted')
        plt.title("LSTM: Actual vs Predicted (first 100 test points)")
        plt.legend()
        plt.savefig("lstm_actual_vs_pred.png")
        plt.close()

    evaluate_lstm(model_lstm, X_test, y_test)

    # Save
    model_lstm.save("improved_lstm_model.keras")
    import joblib
    joblib.dump(scaler_features, "improved_scaler_features.pkl")
    joblib.dump(scaler_target, "improved_scaler_target.pkl")

    ##############################
    # 10) Reinforcement Learning
    ##############################
    class StockTradingEnv(gym.Env):
        """
        Improved RL Env that:
         - excludes raw 'Close' from observation
         - includes transaction cost (optional)
         - uses step-based PnL as reward
        """
        metadata = {'render.modes': ['human']}
        def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
            super().__init__()

            self.df = df.reset_index(drop=True)
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.net_worth = initial_balance
            self.current_step = 0
            self.max_steps = len(df)

            # Add transaction cost in decimal form (0.001 => 0.1%)
            self.transaction_cost = transaction_cost

            self.shares_held = 0
            self.cost_basis = 0

            # Suppose we exclude 'Close' from features to remove direct see of final price
            self.obs_columns = [
                'Open', 'High', 'Low', 'Volume',
                'RSI', 'MACD', 'OBV', 'ADX',
                'BB_Upper', 'BB_Lower', 'BB_Width',
                'MFI', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'STDDEV_5'
            ]

            # We'll normalize features with the same scaler used for LSTM. If you want EXACT same scaling:
            # you can pass the same 'scaler_features' object into this environment.
            self.scaler = MinMaxScaler().fit(df[self.obs_columns])
            # Or load from a pkl if you prefer: joblib.load("improved_scaler_features.pkl")

            self.action_space = spaces.Discrete(3)  # 0=Sell, 1=Hold, 2=Buy
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(len(self.obs_columns) + 3,),  # + balance, shares, cost_basis
                dtype=np.float32
            )

        def reset(self):
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.shares_held = 0
            self.cost_basis = 0
            self.current_step = 0
            return self._get_obs()

        def step(self, action):
            # Current row
            row = self.df.iloc[self.current_step]
            current_price = row['Close']

            prev_net_worth = self.net_worth

            if action == 2:  # Buy
                shares_bought = int(self.balance // current_price)
                if shares_bought > 0:
                    cost = shares_bought * current_price
                    fee = cost * self.transaction_cost
                    self.balance -= (cost + fee)
                    # Weighted average cost basis
                    prev_shares = self.shares_held
                    self.shares_held += shares_bought
                    self.cost_basis = (
                        (self.cost_basis * prev_shares) + (shares_bought * current_price)
                    ) / self.shares_held

            elif action == 0:  # Sell
                if self.shares_held > 0:
                    revenue = self.shares_held * current_price
                    fee = revenue * self.transaction_cost
                    self.balance += (revenue - fee)
                    self.shares_held = 0
                    self.cost_basis = 0

            # Recompute net worth
            self.net_worth = self.balance + self.shares_held * current_price
            self.current_step += 1
            done = (self.current_step >= self.max_steps - 1)

            # *Step-based* reward => daily PnL
            reward = self.net_worth - prev_net_worth

            obs = self._get_obs()
            return obs, reward, done, {}

        def _get_obs(self):
            row = self.df.iloc[self.current_step][self.obs_columns]
            # Scale
            scaled = self.scaler.transform([row])[0]

            additional = np.array([
                self.balance / self.initial_balance,
                self.shares_held / 100.0,
                self.cost_basis / (self.initial_balance+1e-9)
            ], dtype=np.float32)

            obs = np.concatenate([scaled, additional]).astype(np.float32)
            return obs

        def render(self, mode='human'):
            profit = self.net_worth - self.initial_balance
            print(f"Step: {self.current_step}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Shares: {self.shares_held}, "
                  f"NetWorth: {self.net_worth:.2f}, "
                  f"Profit: {profit:.2f}")

    ##############################
    # 11) Train DQN
    ##############################
    def train_dqn(env):
        logging.info("Training DQN agent with improved environment...")
        model = DQN(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.99,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            tensorboard_log="./dqn_enhanced_tensorboard/"
        )
        model.learn(total_timesteps=50000)
        model.save("improved_dqn_agent")
        return model

    # Initialize environment with the same data
    # *In a real scenario, you might feed a different dataset or do a train/test split
    # for the RL environment, too.
    rl_env = StockTradingEnv(data, initial_balance=10000, transaction_cost=0.001)
    vec_env = DummyVecEnv([lambda: rl_env])

    dqn_model = train_dqn(vec_env)
    logging.info("Finished DQN training. You can test with a script like 'use_dqn.py' or do an internal test here.")

if __name__ == "__main__":
    main()

