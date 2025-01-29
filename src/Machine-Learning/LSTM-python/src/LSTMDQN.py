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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

import xgboost as xgb
import optuna
from optuna.integration import KerasPruningCallback

# For Reinforcement Learning
import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Suppress TensorFlow warnings beyond errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###################################################
# 1. Data Loading / Advanced Technical Indicators
###################################################
def load_data(file_path):
    logging.info(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['time'])
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
    df.rename(columns=rename_mapping, inplace=True)

    logging.info(f"Data columns after renaming: {df.columns.tolist()}")
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info("Data loaded and sorted successfully.")
    return df


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    RS = gain / (loss + 1e-9)
    return 100 - (100 / (1 + RS))


def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line  # histogram


def compute_obv(df):
    signed_volume = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    return signed_volume.cumsum()


def compute_adx(df, window=14):
    """Pseudo-ADX approach using rolling True Range / Close."""
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    tr = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    tr_rolling = tr.rolling(window=window).mean()
    adx_placeholder = tr_rolling / (df['Close'] + 1e-9)
    df.drop(['H-L','H-Cp','L-Cp'], axis=1, inplace=True)
    return adx_placeholder


def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / (sma + 1e-9)
    return upper, lower, bandwidth


def compute_mfi(df, window=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    prev_tp = typical_price.shift(1)
    flow_pos = money_flow.where(typical_price > prev_tp, 0)
    flow_neg = money_flow.where(typical_price < prev_tp, 0)
    pos_sum = flow_pos.rolling(window=window).sum()
    neg_sum = flow_neg.rolling(window=window).sum()
    mfi = 100 - (100 / (1 + pos_sum/(neg_sum+1e-9)))
    return mfi


def calculate_technical_indicators(df):
    logging.info("Calculating technical indicators...")

    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['OBV'] = compute_obv(df)
    df['ADX'] = compute_adx(df)

    up, low, bw = compute_bollinger_bands(df['Close'], 20, 2)
    df['BB_Upper'] = up
    df['BB_Lower'] = low
    df['BB_Width'] = bw

    df['MFI'] = compute_mfi(df, 14)

    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['STDDEV_5'] = df['Close'].rolling(5).std()

    df.dropna(inplace=True)
    logging.info("Technical indicators calculated successfully.")
    return df


###############################
# 2. ARGUMENT PARSING
###############################
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train LSTM and DQN models for stock trading.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV data file (with columns time,open,high,low,close,volume).')
    parser.add_argument('--do_dqn_inference', action='store_true',
                        help='If set, will run the DQN inference after training the agent.')
    return parser.parse_args()


###############################
# 3. MAIN
###############################
def main():
    # Parse command-line arguments
    args = parse_arguments()
    csv_path = args.csv_path
    do_dqn_inference = args.do_dqn_inference

    # 1) Load Data
    data = load_data(csv_path)
    data = calculate_technical_indicators(data)

    # We'll exclude 'Close' from the feature set
    feature_columns = [
        'SMA_5','SMA_10','EMA_5','EMA_10','STDDEV_5',
        'RSI','MACD','ADX','OBV','Volume','Open','High','Low',
        'BB_Upper','BB_Lower','BB_Width','MFI'
    ]
    target_column = 'Close'
    data = data[['Date'] + feature_columns + [target_column]]
    data.dropna(inplace=True)

    # 2) Scaling
    scaler_features = MinMaxScaler()
    scaler_target   = MinMaxScaler()

    X_all = data[feature_columns].values
    y_all = data[[target_column]].values

    X_scaled = scaler_features.fit_transform(X_all)
    y_scaled = scaler_target.fit_transform(y_all).flatten()

    # 3) Create LSTM sequences
    def create_sequences(features, target, window_size=15):
        X_seq, y_seq = [], []
        for i in range(len(features)-window_size):
            X_seq.append(features[i:i+window_size])
            y_seq.append(target[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    window_size = 15
    X, y = create_sequences(X_scaled, y_scaled, window_size)

    # 4) Train/Val/Test Split
    train_size = int(len(X)*0.7)
    val_size   = int(len(X)*0.15)
    test_size  = len(X)-train_size-val_size

    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    logging.info(f"Scaled training features shape: {X_train.shape}")
    logging.info(f"Scaled validation features shape: {X_val.shape}")
    logging.info(f"Scaled testing features shape: {X_test.shape}")
    logging.info(f"Scaled training target shape: {y_train.shape}")
    logging.info(f"Scaled validation target shape: {y_val.shape}")
    logging.info(f"Scaled testing target shape: {y_test.shape}")

    # 5) GPU or CPU
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

    # 6) Build LSTM
    def build_advanced_lstm(input_shape, hyperparams):
        model = Sequential()
        for i in range(hyperparams['num_lstm_layers']):
            return_seqs = (i < hyperparams['num_lstm_layers'] - 1)
            model.add(Bidirectional(
                LSTM(hyperparams['lstm_units'],
                     return_sequences=return_seqs,
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                input_shape=input_shape if i==0 else None
            ))
            model.add(Dropout(hyperparams['dropout_rate']))

        model.add(Dense(1, activation='linear'))

        if hyperparams['optimizer'] == 'Adam':
            opt = Adam(learning_rate=hyperparams['learning_rate'], decay=hyperparams['decay'])
        elif hyperparams['optimizer'] == 'Nadam':
            opt = Nadam(learning_rate=hyperparams['learning_rate'])
        else:
            opt = Adam(learning_rate=hyperparams['learning_rate'])

        model.compile(optimizer=opt, loss=Huber(), metrics=['mae'])
        return model

    # 7) Optuna Tuning
    def objective(trial):
        num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
        lstm_units      = trial.suggest_categorical('lstm_units', [32,64,96,128])
        dropout_rate    = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate   = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        optimizer_name  = trial.suggest_categorical('optimizer', ['Adam','Nadam'])
        decay           = trial.suggest_float('decay', 0.0, 1e-4)

        hyperparams = {
            'num_lstm_layers': num_lstm_layers,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'decay': decay
        }

        model_ = build_advanced_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_reduce  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        cb_prune   = KerasPruningCallback(trial, 'val_loss')

        history = model_.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, lr_reduce, cb_prune],
            verbose=0
        )
        val_mae = min(history.history['val_mae'])
        return val_mae

    logging.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    logging.info(f"Best Hyperparameters from Optuna: {best_params}")

    # 8) Train the Best LSTM
    best_model = build_advanced_lstm((X_train.shape[1], X_train.shape[2]), best_params)

    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_reduce_final  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    logging.info("Training the best LSTM model with optimized hyperparameters...")
    history = best_model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_final, lr_reduce_final],
        verbose=1
    )

    # 9) Evaluate
    def evaluate_model(model, X_test, y_test):
        logging.info("Evaluating model (LSTM)...")
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1,1)).flatten()

        mse = mean_squared_error(y_test_actual, y_pred)
        rmse= np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2  = r2_score(y_test_actual, y_pred)

        direction_actual = np.sign(np.diff(y_test_actual))
        direction_pred   = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(direction_actual==direction_pred)

        logging.info(f"Test MSE: {mse}")
        logging.info(f"Test RMSE: {rmse}")
        logging.info(f"Test MAE: {mae}")
        logging.info(f"Test R2 Score: {r2}")
        logging.info(f"Directional Accuracy: {directional_accuracy}")

        # Plot
        plt.figure(figsize=(14,7))
        plt.plot(y_test_actual, label='Actual Price')
        plt.plot(y_pred,        label='Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('actual_vs_predicted.png')
        plt.close()
        logging.info("Plot saved as 'actual_vs_predicted.png'")

        # Tabulate first 40
        table_data = []
        for i in range(min(40, len(y_test_actual))):
            table_data.append([i, round(y_test_actual[i],2), round(y_pred[i],2)])
        headers = ["Index", "Actual Price", "Predicted Price"]
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

        return mse, rmse, mae, r2, directional_accuracy

    mse, rmse, mae, r2, directional_accuracy = evaluate_model(best_model, X_test, y_test)

    # 10) Save
    best_model.save('optimized_lstm_model.h5')
    import joblib
    joblib.dump(scaler_features, 'scaler_features.save')
    joblib.dump(scaler_target, 'scaler_target.save')
    logging.info("Model and scalers saved (optimized_lstm_model.h5, scaler_features.save, scaler_target.save).")

    ##########################################################
    # 11) Reinforcement Learning: StockTradingEnv + DQN
    ##########################################################
    class StockTradingEnv(gym.Env):
        """
        A simple stock trading environment for OpenAI Gym
        with step-based reward = net_worth - initial_balance.
        """
        metadata = {'render.modes': ['human']}

        def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
            super().__init__()
            self.df = df.reset_index()
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.net_worth = initial_balance
            self.max_steps = len(df)
            self.current_step = 0
            self.shares_held = 0
            self.cost_basis = 0
            self.transaction_cost = transaction_cost

            # Re-use feature_columns from above
            self.feature_columns = feature_columns

            # Action space: 0=Sell,1=Hold,2=Buy
            self.action_space = spaces.Discrete(3)

            # Observation = [17 indicators + balance + shares + cost_basis] => total 20
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(len(self.feature_columns)+3,),
                dtype=np.float32
            )

        def reset(self):
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.current_step = 0
            self.shares_held = 0
            self.cost_basis = 0
            return self._next_observation()

        def _next_observation(self):
            obs_vals = self.df.loc[self.current_step, self.feature_columns].values
            # simple normalization
            if np.max(obs_vals)!=0:
                obs_vals = obs_vals / np.max(obs_vals)

            additional = np.array([
                self.balance/self.initial_balance,
                self.shares_held/100.0,
                self.cost_basis/self.initial_balance
            ], dtype=np.float32)

            return np.concatenate([obs_vals, additional]).astype(np.float32)

        def step(self, action):
            current_price = self.df.loc[self.current_step, 'Close']

            if action==2:  # Buy
                shares_bought = int(self.balance // current_price)
                if shares_bought>0:
                    cost= shares_bought* current_price
                    fee = cost* self.transaction_cost
                    self.balance-= (cost+ fee)
                    old_shares= self.shares_held
                    self.shares_held+= shares_bought
                    # Weighted average cost
                    self.cost_basis=(
                        (self.cost_basis* old_shares)+(shares_bought* current_price)
                    )/ self.shares_held

            elif action==0: # Sell
                if self.shares_held>0:
                    revenue= self.shares_held* current_price
                    fee = revenue*self.transaction_cost
                    self.balance+= (revenue- fee)
                    self.shares_held=0
                    self.cost_basis=0

            prev_net_worth= self.net_worth
            self.net_worth= self.balance+ self.shares_held* current_price
            self.current_step+=1
            done= (self.current_step>= self.max_steps-1)

            # Reward: net_worth - initial_balance (like original code)
            reward= self.net_worth- self.initial_balance

            obs= self._next_observation()
            return obs, reward, done, {}

        def render(self, mode='human'):
            profit= self.net_worth- self.initial_balance
            print(f"Step: {self.current_step}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Shares: {self.shares_held}, "
                  f"NetWorth: {self.net_worth:.2f}, "
                  f"Profit: {profit:.2f}")


    def train_dqn_agent(env):
        logging.info("Training DQN Agent (step-based reward).")
        try:
            model = DQN(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=64,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                tensorboard_log="./dqn_stock_tensorboard/"
            )
            model.learn(total_timesteps=100000)
            model.save("dqn_stock_trading")
            logging.info("DQN Agent trained and saved as 'dqn_stock_trading.zip'.")
            return model
        except Exception as e:
            logging.error(f"Error training DQN Agent: {e}")
            sys.exit(1)


    # 12) Train the DQN environment
    logging.info("Initializing trading environment for DQN training...")
    trading_env = StockTradingEnv(data, initial_balance=10000, transaction_cost=0.001)
    vec_env = DummyVecEnv([lambda: trading_env])

    dqn_model = train_dqn_agent(vec_env)
    logging.info("DQN training complete.")

    # 13) Optional: run DQN inference right away (like use_dqn.py) if user wants
    if do_dqn_inference:
        logging.info("Running DQN inference (test) after training...")
        obs = vec_env.reset()
        done = [False]
        total_reward = 0.0
        step_data = []
        step_count = 0

        # underlying env to access net worth, etc.
        underlying_env = vec_env.envs[0]

        while not done[0]:
            step_count += 1
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            reward_scalar = reward[0]
            total_reward += reward_scalar

            step_data.append({
                "Step": step_count,
                "Action": int(action[0]),
                "Reward": reward_scalar,
                "Balance": underlying_env.balance,
                "Shares": underlying_env.shares_held,
                "NetWorth": underlying_env.net_worth
            })

        final_net_worth = underlying_env.net_worth
        final_profit = final_net_worth - underlying_env.initial_balance

        print("\n=== DQN Agent Finished ===")
        print(f"Total Steps Taken: {step_count}")
        print(f"Final Net Worth: {final_net_worth:.2f}")
        print(f"Final Profit: {final_profit:.2f}")
        print(f"Sum of Rewards: {total_reward:.2f}")

        buy_count  = sum(1 for x in step_data if x["Action"] == 2)
        sell_count = sum(1 for x in step_data if x["Action"] == 0)
        hold_count = sum(1 for x in step_data if x["Action"] == 1)
        print(f"Actions Taken -> BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

        # Show last 15 steps (like use_dqn)
        steps_to_display = 15
        last_n = step_data[-steps_to_display:] if len(step_data)> steps_to_display else step_data
        rows = []
        for d in last_n:
            rows.append([
                d["Step"], d["Action"], f"{d['Reward']:.2f}",
                f"{d['Balance']:.2f}", d["Shares"], f"{d['NetWorth']:.2f}"
            ])
        headers = ["Step","Action","Reward","Balance","Shares","NetWorth"]
        print(f"\n== Last {steps_to_display} Steps ==")
        print(tabulate(rows, headers=headers, tablefmt="pretty"))

    logging.info("All tasks complete. Exiting.")


if __name__ == "__main__":
    main()

