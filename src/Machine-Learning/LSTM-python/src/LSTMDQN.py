import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import GPUtil
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import optuna
from optuna.integration import KerasPruningCallback

import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from multiprocessing import Pool, cpu_count
import threading
import time

# Suppress TensorFlow logs beyond errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================
# Resource Detection Functions
# ============================
def get_cpu_info():
    """
    Retrieves CPU information including physical and logical cores and current usage per core.
    
    Returns:
        dict: Dictionary containing physical cores, logical cores, and CPU usage per core.
    """
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    return {
        'physical_cores': cpu_count,
        'logical_cores': cpu_count_logical,
        'cpu_percent': cpu_percent
    }

def get_gpu_info():
    """
    Retrieves GPU information including load, memory usage, and temperature.
    
    Returns:
        list: List of dictionaries containing GPU stats.
    """
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'load': gpu.load * 100,  # Convert to percentage
            'memory_total': gpu.memoryTotal,
            'memory_used': gpu.memoryUsed,
            'memory_free': gpu.memoryFree,
            'temperature': gpu.temperature
        })
    return gpu_info

def configure_tensorflow(cpu_stats, gpu_stats):
    """
    Configures TensorFlow to utilize available CPU and GPU resources efficiently.
    
    Args:
        cpu_stats (dict): Dictionary containing CPU statistics.
        gpu_stats (list): List of dictionaries containing GPU statistics.
    """
    logical_cores = cpu_stats['logical_cores']
    os.environ["OMP_NUM_THREADS"] = str(logical_cores)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(logical_cores)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(logical_cores)
    
    if gpu_stats:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                logging.error(f"TensorFlow GPU configuration error: {e}")
    else:
        tf.config.threading.set_intra_op_parallelism_threads(logical_cores)
        tf.config.threading.set_inter_op_parallelism_threads(logical_cores)
        logging.info("Configured TensorFlow to use CPU with optimized thread settings.")

# ============================
# Resource Monitoring Function (Optional)
# ============================
def monitor_resources(interval=60):
    """
    Continuously monitors and logs CPU and GPU usage at specified intervals.
    
    Args:
        interval (int): Time in seconds between each monitoring snapshot.
    """
    while True:
        cpu = psutil.cpu_percent(interval=1, percpu=True)
        gpu = get_gpu_info()
        logging.info(f"CPU Usage per Core: {cpu}%")
        if gpu:
            for gpu_stat in gpu:
                logging.info(f"GPU {gpu_stat['id']} - {gpu_stat['name']}: Load: {gpu_stat['load']}%, "
                             f"Memory Used: {gpu_stat['memory_used']}MB / {gpu_stat['memory_total']}MB, "
                             f"Temperature: {gpu_stat['temperature']}°C")
        else:
            logging.info("No GPUs detected.")
        logging.info("-" * 50)
        time.sleep(interval)

# ============================
# Data Loading & Technical Indicators
# ============================
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
    ema_long  = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line  # histogram

def compute_obv(df):
    signed_volume = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    return signed_volume.cumsum()

def compute_adx(df, window=14):
    df['H-L']  = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low']  - df['Close'].shift(1)).abs()
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
    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-9)))
    return mfi

def calculate_technical_indicators(df):
    logging.info("Calculating technical indicators...")
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['OBV'] = compute_obv(df)
    df['ADX'] = compute_adx(df)

    up, lo, bw = compute_bollinger_bands(df['Close'], 20, 2)
    df['BB_Upper'] = up
    df['BB_Lower'] = lo
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

# ============================
# Argument Parsing
# ============================
def parse_arguments():
    parser = argparse.ArgumentParser(description='All-in-One: LSTM + DQN (with LSTM predictions) + Tuning.')
    parser.add_argument('csv_path', type=str,
                        help='Path to CSV data with columns [time, open, high, low, close, volume].')
    parser.add_argument('--lstm_window_size', type=int, default=15,
                        help='Sequence window size for LSTM. Default=15.')
    parser.add_argument('--dqn_total_timesteps', type=int, default=50000,
                        help='Total timesteps to train each DQN candidate. Default=50000.')
    parser.add_argument('--dqn_eval_episodes', type=int, default=1,
                        help='Number of episodes to evaluate DQN in the tuning step. Default=1 (entire dataset once).')
    parser.add_argument('--n_trials_lstm', type=int, default=30,
                        help='Number of Optuna trials for LSTM. Default=30.')
    parser.add_argument('--n_trials_dqn', type=int, default=20,
                        help='Number of Optuna trials for DQN. Default=20.')
    parser.add_argument('--max_parallel_trials', type=int, default=None,
                        help='Maximum number of parallel Optuna trials. Defaults to (logical cores - 2).')
    parser.add_argument('--preprocess_workers', type=int, default=None,
                        help='Number of worker processes for data preprocessing. Defaults to (logical cores - 2).')
    parser.add_argument('--monitor_resources', action='store_true',
                        help='Enable real-time resource monitoring.')
    return parser.parse_args()

# ============================
# Custom DQN Callback: Log Actions + Rewards
# ============================
class ActionLoggingCallback(BaseCallback):
    """
    Logs distribution of actions and average reward after each rollout.
    For off-policy (DQN), "rollout" can be a bit different than on-policy,
    but stable-baselines3 still calls `_on_rollout_end` periodically.
    """
    def __init__(self, verbose=0):
        super(ActionLoggingCallback, self).__init__(verbose)
        self.action_buffer = []
        self.reward_buffer = []

    def _on_training_start(self):
        self.action_buffer = []
        self.reward_buffer = []

    def _on_step(self):
        # For Stable Baselines3, access actions and rewards via self.locals
        action = self.locals.get('action', None)
        reward = self.locals.get('reward', None)
        if action is not None:
            self.action_buffer.append(action)
        if reward is not None:
            self.reward_buffer.append(reward)
        return True

    def _on_rollout_end(self):
        import numpy as np
        actions = np.array(self.action_buffer)
        rewards = np.array(self.reward_buffer)
        if len(actions) > 0:
            unique, counts = np.unique(actions, return_counts=True)
            total = len(actions)
            distr_str = []
            for act, c in zip(unique, counts):
                distr_str.append(f"Action {act}: {c} times ({100 * c / total:.2f}%)")
            logging.info(" -- DQN Rollout End -- ")
            logging.info("    " + ", ".join(distr_str))
            logging.info(f"    Avg Reward this rollout: {rewards.mean():.4f} (min={rewards.min():.4f}, max={rewards.max():.4f})")
        self.action_buffer = []
        self.reward_buffer = []

# ============================
# Data Preprocessing with Controlled Parallelization
# ============================
def parallel_feature_engineering(row):
    """
    Placeholder function for feature engineering. Modify as needed.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        pd.Series: Processed row.
    """
    # Implement any additional feature engineering here if necessary
    return row

def feature_engineering_parallel(df, num_workers):
    """
    Applies feature engineering in parallel using multiprocessing.
    
    Args:
        df (pd.DataFrame): DataFrame to process.
        num_workers (int): Number of worker processes.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    logging.info(f"Starting parallel feature engineering with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        processed_rows = pool.map(parallel_feature_engineering, [row for _, row in df.iterrows()])
    df_processed = pd.DataFrame(processed_rows)
    logging.info("Parallel feature engineering completed.")
    return df_processed

# ============================
# Main Function with Enhanced Optimizations
# ============================
def main():
    args = parse_arguments()
    csv_path = args.csv_path
    lstm_window_size = args.lstm_window_size
    dqn_total_timesteps = args.dqn_total_timesteps
    dqn_eval_episodes   = args.dqn_eval_episodes
    n_trials_lstm = args.n_trials_lstm
    n_trials_dqn  = args.n_trials_dqn
    max_parallel_trials = args.max_parallel_trials
    preprocess_workers = args.preprocess_workers
    enable_resource_monitor = args.monitor_resources

    # =============================
    # Setup Logging
    # =============================
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("LSTMDQN.log"),
                            logging.StreamHandler(sys.stdout)
                        ])

    # =============================
    # Resource Detection & Logging
    # =============================
    cpu_stats = get_cpu_info()
    gpu_stats = get_gpu_info()

    logging.info("===== Resource Statistics =====")
    logging.info(f"Physical CPU Cores: {cpu_stats['physical_cores']}")
    logging.info(f"Logical CPU Cores: {cpu_stats['logical_cores']}")
    logging.info(f"CPU Usage per Core: {cpu_stats['cpu_percent']}%")

    if gpu_stats:
        logging.info("GPU Statistics:")
        for gpu in gpu_stats:
            logging.info(f"GPU {gpu['id']} - {gpu['name']}: Load: {gpu['load']}%, "
                         f"Memory Used: {gpu['memory_used']}MB / {gpu['memory_total']}MB, "
                         f"Temperature: {gpu['temperature']}°C")
    else:
        logging.info("No GPUs detected.")
    logging.info("=================================")

    # =============================
    # Configure TensorFlow
    # =============================
    configure_tensorflow(cpu_stats, gpu_stats)

    # =============================
    # Start Resource Monitoring (Optional)
    # =============================
    if enable_resource_monitor:
        logging.info("Starting real-time resource monitoring...")
        resource_monitor_thread = threading.Thread(target=monitor_resources, args=(60,), daemon=True)
        resource_monitor_thread.start()

    ##########################################
    # A) LSTM PART: LOAD, PREPROCESS, TUNE
    ##########################################
    # 1) LOAD & preprocess
    df = load_data(csv_path)
    df = calculate_technical_indicators(df)

    feature_columns = [
        'SMA_5','SMA_10','EMA_5','EMA_10','STDDEV_5',
        'RSI','MACD','ADX','OBV','Volume','Open','High','Low',
        'BB_Upper','BB_Lower','BB_Width','MFI'
    ]
    target_column = 'Close'
    df = df[['Date'] + feature_columns + [target_column]].dropna()

    # 2) Controlled Parallel Data Preprocessing
    if preprocess_workers is None:
        # Default to logical cores minus 2 to prevent overloading
        preprocess_workers = max(1, cpu_stats['logical_cores'] - 2)
    else:
        preprocess_workers = min(preprocess_workers, cpu_stats['logical_cores'])

    df = feature_engineering_parallel(df, num_workers=preprocess_workers)

    scaler_features = MinMaxScaler()
    scaler_target   = MinMaxScaler()

    X_all = df[feature_columns].values
    y_all = df[[target_column]].values

    X_scaled = scaler_features.fit_transform(X_all)
    y_scaled = scaler_target.fit_transform(y_all).flatten()

    # 3) Create sequences
    def create_sequences(features, target, window_size):
        X_seq, y_seq = [], []
        for i in range(len(features) - window_size):
            X_seq.append(features[i:i+window_size])
            y_seq.append(target[i+window_size])
        return np.array(X_seq), np.array(y_seq)

    X, y = create_sequences(X_scaled, y_scaled, lstm_window_size)

    # 4) Split into train/val/test
    train_size = int(len(X) * 0.7)
    val_size   = int(len(X) * 0.15)
    test_size  = len(X) - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val,   y_val   = X[train_size: train_size + val_size], y[train_size: train_size + val_size]
    X_test,  y_test  = X[train_size + val_size:], y[train_size + val_size:]

    logging.info(f"Scaled training features shape: {X_train.shape}")
    logging.info(f"Scaled validation features shape: {X_val.shape}")
    logging.info(f"Scaled testing features shape: {X_test.shape}")
    logging.info(f"Scaled training target shape: {y_train.shape}")
    logging.info(f"Scaled validation target shape: {y_val.shape}")
    logging.info(f"Scaled testing target shape: {y_test.shape}")

    # 5) Build and compile LSTM model
    def build_lstm(input_shape, hyperparams):
        model = Sequential()
        num_layers = hyperparams['num_lstm_layers']
        units      = hyperparams['lstm_units']
        drop       = hyperparams['dropout_rate']
        for i in range(num_layers):
            return_seqs = (i < num_layers - 1)
            model.add(Bidirectional(
                LSTM(units, return_sequences=return_seqs, kernel_regularizer=l2(1e-4)),
                input_shape=input_shape if i == 0 else None
            ))
            model.add(Dropout(drop))
        model.add(Dense(1, activation='linear'))

        opt_name = hyperparams['optimizer']
        lr       = hyperparams['learning_rate']
        decay    = hyperparams['decay']
        if opt_name == 'Adam':
            opt = Adam(learning_rate=lr, decay=decay)
        elif opt_name == 'Nadam':
            opt = Nadam(learning_rate=lr)
        else:
            opt = Adam(learning_rate=lr)

        model.compile(loss=Huber(), optimizer=opt, metrics=['mae'])
        return model

    # 6) Optuna objective for LSTM
    def lstm_objective(trial):
        num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
        lstm_units      = trial.suggest_categorical('lstm_units', [32, 64, 96, 128])
        dropout_rate    = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate   = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        optimizer_name  = trial.suggest_categorical('optimizer', ['Adam', 'Nadam'])
        decay           = trial.suggest_float('decay', 0.0, 1e-4)

        hyperparams = {
            'num_lstm_layers': num_lstm_layers,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'optimizer': optimizer_name,
            'decay': decay
        }

        model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
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

    # 7) Hyperparameter Optimization with Optuna (Parallelized)
    if max_parallel_trials is None:
        # Default to logical cores minus 2 to prevent overloading
        max_parallel_trials = max(1, cpu_stats['logical_cores'] - 2)
    else:
        max_parallel_trials = min(max_parallel_trials, cpu_stats['logical_cores'])

    logging.info(f"Starting LSTM hyperparameter optimization with Optuna using {max_parallel_trials} parallel trials...")
    study_lstm = optuna.create_study(direction='minimize')
    study_lstm.optimize(lstm_objective, n_trials=n_trials_lstm, n_jobs=max_parallel_trials)
    best_lstm_params = study_lstm.best_params
    logging.info(f"Best LSTM Hyperparameters: {best_lstm_params}")

    # 8) Train final LSTM
    final_lstm = build_lstm((X_train.shape[1], X_train.shape[2]), best_lstm_params)
    early_stop_final = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_reduce_final  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    logging.info("Training best LSTM model with optimized hyperparameters...")
    hist = final_lstm.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_final, lr_reduce_final],
        verbose=1
    )

    # 9) Evaluate LSTM
    def evaluate_lstm(model, X_test, y_test):
        logging.info("Evaluating final LSTM model...")
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mse_ = mean_squared_error(y_test_actual, y_pred)
        rmse_ = np.sqrt(mse_)
        mae_ = mean_absolute_error(y_test_actual, y_pred)
        r2_  = r2_score(y_test_actual, y_pred)

        direction_actual = np.sign(np.diff(y_test_actual))
        direction_pred  = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(direction_actual == direction_pred)

        logging.info(f"Test MSE: {mse_:.4f}")
        logging.info(f"Test RMSE: {rmse_:.4f}")
        logging.info(f"Test MAE: {mae_:.4f}")
        logging.info(f"Test R2 Score: {r2_:.4f}")
        logging.info(f"Directional Accuracy: {directional_accuracy:.4f}")

        # Plot Actual vs Predicted
        plt.figure(figsize=(14, 7))
        plt.plot(y_test_actual, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price')
        plt.title('LSTM: Actual vs Predicted Closing Prices')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstm_actual_vs_pred.png')
        plt.close()

        # Tabulate first 40 results
        table = []
        limit = min(40, len(y_test_actual))
        for i in range(limit):
            table.append([i, round(y_test_actual[i], 2), round(y_pred[i], 2)])
        headers = ["Index", "Actual Price", "Predicted Price"]
        print("\nFirst 40 Actual vs. Predicted Prices:")
        print(tabulate(table, headers=headers, tablefmt="pretty"))
        return r2_, directional_accuracy

    _r2, _diracc = evaluate_lstm(final_lstm, X_test, y_test)

    # 10) Save LSTM and Scalers
    final_lstm.save('best_lstm_model.h5')
    joblib.dump(scaler_features, 'scaler_features.pkl')
    joblib.dump(scaler_target, 'scaler_target.pkl')
    logging.info("Saved best LSTM model and scaler objects (best_lstm_model.h5, scaler_features.pkl, scaler_target.pkl).")

    ############################################################
    # B) DQN PART: BUILD ENV THAT USES THE LSTM + FORECAST
    ############################################################
    class StockTradingEnvWithLSTM(gym.Env):
        """
        A custom OpenAI Gym environment for stock trading that integrates LSTM model predictions.
        Observation includes technical indicators, account information, and predicted next close price.
        """
        metadata = {'render.modes': ['human']}

        def __init__(self, df, feature_columns, lstm_model, scaler_features, scaler_target,
                     window_size=15, initial_balance=10000, transaction_cost=0.001):
            super(StockTradingEnvWithLSTM, self).__init__()
            self.df = df.reset_index(drop=True)
            self.feature_columns = feature_columns
            self.lstm_model = lstm_model
            self.scaler_features = scaler_features
            self.scaler_target = scaler_target
            self.window_size = window_size

            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.net_worth = initial_balance
            self.transaction_cost = transaction_cost

            self.max_steps = len(df)
            self.current_step = 0
            self.shares_held = 0
            self.cost_basis = 0

            # Raw array of features
            self.raw_features = df[feature_columns].values

            # Action space: 0=Sell, 1=Hold, 2=Buy
            self.action_space = spaces.Discrete(3)

            # Observation space: [technical indicators, balance, shares, cost_basis, predicted_next_close]
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(len(feature_columns) + 3 + 1,),
                dtype=np.float32
            )

        def reset(self):
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.current_step = 0
            self.shares_held = 0
            self.cost_basis = 0
            return self._get_obs()

        def _get_obs(self):
            row = self.raw_features[self.current_step]
            row_max = np.max(row) if np.max(row) != 0 else 1.0
            row_norm = row / row_max

            # Account info
            additional = np.array([
                self.balance / self.initial_balance,
                self.shares_held / 100.0,  # Assuming max 100 shares for normalization
                self.cost_basis / (self.initial_balance + 1e-9)
            ], dtype=np.float32)

            # LSTM prediction
            if self.current_step < self.window_size:
                # Not enough history => no forecast
                predicted_close = 0.0
            else:
                seq = self.raw_features[self.current_step - self.window_size: self.current_step]
                seq_scaled = self.scaler_features.transform(seq)
                seq_scaled = np.expand_dims(seq_scaled, axis=0)  # shape (1, window_size, #features)
                pred_scaled = self.lstm_model.predict(seq_scaled, verbose=0).flatten()[0]
                pred_scaled = np.clip(pred_scaled, 0, 1)
                unscaled = self.scaler_target.inverse_transform([[pred_scaled]])[0, 0]
                # Normalize predicted close price (assuming a typical price range)
                predicted_close = unscaled / 1000.0

            obs = np.concatenate([row_norm, additional, [predicted_close]]).astype(np.float32)
            return obs

        def step(self, action):
            prev_net_worth = self.net_worth
            current_price = self.df.loc[self.current_step, 'Close']

            if action == 2:  # BUY
                shares_bought = int(self.balance // current_price)
                if shares_bought > 0:
                    cost = shares_bought * current_price
                    fee = cost * self.transaction_cost
                    self.balance -= (cost + fee)
                    old_shares = self.shares_held
                    self.shares_held += shares_bought
                    self.cost_basis = (
                        (self.cost_basis * old_shares) + (shares_bought * current_price)
                    ) / self.shares_held

            elif action == 0:  # SELL
                if self.shares_held > 0:
                    revenue = self.shares_held * current_price
                    fee = revenue * self.transaction_cost
                    self.balance += (revenue - fee)
                    self.shares_held = 0
                    self.cost_basis = 0

            self.net_worth = self.balance + self.shares_held * current_price
            self.current_step += 1
            done = (self.current_step >= self.max_steps - 1)

            reward = self.net_worth - self.initial_balance
            obs = self._get_obs()
            return obs, reward, done, {}

        def render(self, mode='human'):
            profit = self.net_worth - self.initial_balance
            print(f"Step: {self.current_step}, "
                  f"Balance={self.balance:.2f}, "
                  f"Shares={self.shares_held}, "
                  f"NetWorth={self.net_worth:.2f}, "
                  f"Profit={profit:.2f}")

    ###################################
    # C) DQN HYPERPARAMETER TUNING WITH LSTM
    ###################################
    from stable_baselines3.common.evaluation import evaluate_policy

    def evaluate_dqn_networth(model, env, n_episodes=1):
        """
        Evaluates the trained DQN model by simulating trading over a specified number of episodes.
        
        Args:
            model (stable_baselines3.DQN): Trained DQN model.
            env (gym.Env): Trading environment instance.
            n_episodes (int): Number of episodes to run for evaluation.
        
        Returns:
            float: Average final net worth across episodes.
        """
        final_net_worths = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            final_net_worths.append(env.net_worth)
        return np.mean(final_net_worths)

    def dqn_objective(trial):
        """
        Objective function for Optuna to optimize DQN hyperparameters.
        Minimizes the negative of the final net worth achieved by the DQN agent.
        
        Args:
            trial (optuna.trial.Trial): Optuna trial object.
        
        Returns:
            float: Negative of the final net worth.
        """
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        gamma = trial.suggest_float("gamma", 0.8, 0.9999)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.01, 0.3)
        buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000])
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        # Initialize environment
        env = StockTradingEnvWithLSTM(
            df=df,
            feature_columns=feature_columns,
            lstm_model=final_lstm,   # Use the trained LSTM model
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            window_size=lstm_window_size
        )
        vec_env = DummyVecEnv([lambda: env])

        # Initialize DQN model
        dqn_action_logger = ActionLoggingCallback(verbose=0)

        model = DQN(
            'MlpPolicy',
            vec_env,
            verbose=0,
            learning_rate=lr,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            buffer_size=buffer_size,
            batch_size=batch_size,
            train_freq=4,
            target_update_interval=1000
        )

        # Train DQN model
        model.learn(total_timesteps=dqn_total_timesteps, callback=dqn_action_logger)

        # Evaluate final net worth
        final_net_worth = evaluate_dqn_networth(model, env, n_episodes=dqn_eval_episodes)
        # Objective is to maximize net worth, so return negative
        return -final_net_worth

    # 11) Hyperparameter Optimization with Optuna (Parallelized)
    if max_parallel_trials is None:
        # Default to logical cores minus 2 to prevent overloading
        max_parallel_trials = max(1, cpu_stats['logical_cores'] - 2)
    else:
        max_parallel_trials = min(max_parallel_trials, cpu_stats['logical_cores'])

    logging.info(f"Starting DQN hyperparameter tuning with Optuna using {max_parallel_trials} parallel trials...")
    study_dqn = optuna.create_study(direction='minimize')
    study_dqn.optimize(dqn_objective, n_trials=n_trials_dqn, n_jobs=max_parallel_trials)
    best_dqn_params = study_dqn.best_params
    logging.info(f"Best DQN Hyperparameters: {best_dqn_params}")

    ###################################
    # D) TRAIN FINAL DQN WITH BEST PARAMETERS
    ###################################
    logging.info("Training final DQN model with best hyperparameters...")
    env_final = StockTradingEnvWithLSTM(
        df=df,
        feature_columns=feature_columns,
        lstm_model=final_lstm,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        window_size=lstm_window_size
    )
    vec_env_final = DummyVecEnv([lambda: env_final])

    final_dqn_logger = ActionLoggingCallback(verbose=1)  # Enable detailed logging

    final_model = DQN(
        'MlpPolicy',
        vec_env_final,
        verbose=1,
        learning_rate=best_dqn_params['lr'],
        gamma=best_dqn_params['gamma'],
        exploration_fraction=best_dqn_params['exploration_fraction'],
        buffer_size=best_dqn_params['buffer_size'],
        batch_size=best_dqn_params['batch_size'],
        train_freq=4,
        target_update_interval=1000
    )
    final_model.learn(total_timesteps=dqn_total_timesteps, callback=final_dqn_logger)
    final_model.save("best_dqn_model_lstm.zip")
    logging.info("Final DQN model trained and saved as 'best_dqn_model_lstm.zip'.")

    ###################################
    # E) FINAL INFERENCE & LOG RESULTS
    ###################################
    logging.info("Running final inference with the trained DQN model...")

    env_test = StockTradingEnvWithLSTM(
        df=df,
        feature_columns=feature_columns,
        lstm_model=final_lstm,
        scaler_features=scaler_features,
        scaler_target=scaler_target,
        window_size=lstm_window_size
    )
    obs = env_test.reset()
    done = False
    total_reward = 0.0
    step_data = []
    step_count = 0

    while not done:
        step_count += 1
        action, _ = final_model.predict(obs, deterministic=True)
        obs, reward, done, info = env_test.step(action)
        total_reward += reward
        step_data.append({
            "Step": step_count,
            "Action": int(action),
            "Reward": reward,
            "Balance": env_test.balance,
            "Shares": env_test.shares_held,
            "NetWorth": env_test.net_worth
        })

    final_net_worth = env_test.net_worth
    final_profit = final_net_worth - env_test.initial_balance

    print("\n=== Final DQN Inference ===")
    print(f"Total Steps: {step_count}")
    print(f"Final Net Worth: {final_net_worth:.2f}")
    print(f"Final Profit: {final_profit:.2f}")
    print(f"Sum of Rewards: {total_reward:.2f}")

    buy_count  = sum(1 for x in step_data if x["Action"] == 2)
    sell_count = sum(1 for x in step_data if x["Action"] == 0)
    hold_count = sum(1 for x in step_data if x["Action"] == 1)
    print(f"Actions Taken -> BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

    # Show last 15 steps
    last_n = step_data[-15:] if len(step_data) > 15 else step_data
    rows = []
    for d in last_n:
        rows.append([
            d["Step"],
            d["Action"],
            f"{d['Reward']:.2f}",
            f"{d['Balance']:.2f}",
            d["Shares"],
            f"{d['NetWorth']:.2f}"
        ])
    headers = ["Step", "Action", "Reward", "Balance", "Shares", "NetWorth"]
    print(f"\n== Last 15 Steps ==")
    print(tabulate(rows, headers=headers, tablefmt="pretty"))

    logging.info("Final inference completed. Results logged and displayed.")

    ###################################
    # F) OPTIONAL: RETRY LOOP IF NET WORTH < THRESHOLD
    ###################################
    NET_WORTH_THRESHOLD = 10500.0  # example threshold

    if final_net_worth < NET_WORTH_THRESHOLD:
        logging.warning(f"Final net worth (${final_net_worth:.2f}) is below ${NET_WORTH_THRESHOLD:.2f}. Retraining the same DQN model to learn from mistakes...")

        # We continue training the SAME final_model without resetting its replay buffer.
        # By setting `reset_num_timesteps=False`, we keep the replay buffer and learned weights.
        additional_timesteps = 50000
        logging.info(f"Retraining the existing DQN model for an additional {additional_timesteps} timesteps (keeping old experiences).")

        # If you want to see action distributions again, you can keep the same callback or define a new one:
        final_model.learn(
            total_timesteps=additional_timesteps, 
            reset_num_timesteps=False,       # Keep replay buffer + internal step counter
            callback=final_dqn_logger        # Optional: to log actions again
        )

        # Evaluate again
        obs = env_test.reset()
        done = False
        second_total_reward = 0.0
        while not done:
            action, _ = final_model.predict(obs, deterministic=True)
            obs, reward, done, info = env_test.step(action)
            second_total_reward += reward

        second_net_worth = env_test.net_worth
        second_profit = second_net_worth - env_test.initial_balance
        logging.info(f"After additional training, new final net worth=${second_net_worth:.2f}, profit=${second_profit:.2f}")

        if second_net_worth < NET_WORTH_THRESHOLD:
            logging.warning("Even after continued training, net worth is still below threshold. Consider a deeper hyperparameter search or analyzing the environment settings.")


main()

