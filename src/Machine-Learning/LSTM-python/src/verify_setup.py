import os
import sys
import pandas as pd
import tensorflow as tf
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import numpy as np
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockTradingEnv(gym.Env):
    """
    A minimal stock trading environment for testing DummyVecEnv.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_steps = len(df)
        self.current_step = 0
        self.shares_held = 0
        self.cost_basis = 0

        # Actions: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # Observations: [normalized features + balance + shares held + cost basis]
        # For simplicity, we'll use the same features as your main script
        feature_columns = ['SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'STDDEV_5', 'RSI', 'MACD', 'ADX', 'OBV', 'Volume', 'Open', 'High', 'Low']
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(feature_columns) + 3,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = 0
        self.shares_held = 0
        self.cost_basis = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ['SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'STDDEV_5', 'RSI', 'MACD', 'ADX', 'OBV', 'Volume', 'Open', 'High', 'Low']].values
        # Normalize additional features
        additional = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100,  # Assuming a maximum of 100 shares for normalization
            self.cost_basis / self.initial_balance
        ])
        return np.concatenate([obs, additional])

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        if action == 2:  # Buy
            total_possible = self.balance // current_price
            shares_bought = total_possible
            if shares_bought > 0:
                self.balance -= shares_bought * current_price
                self.shares_held += shares_bought
                self.cost_basis = (self.cost_basis * (self.shares_held - shares_bought) + shares_bought * current_price) / self.shares_held
        elif action == 0:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.cost_basis = 0
        # Hold does nothing

        self.net_worth = self.balance + self.shares_held * current_price
        self.current_step += 1

        done = self.current_step >= self.max_steps - 1

        # Reward: change in net worth
        reward = self.net_worth - self.initial_balance

        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Cost Basis: {self.cost_basis})')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')

def main(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    logging.info("File exists.")

    # Load a small portion of the data
    try:
        data = pd.read_csv(file_path, nrows=5)
        logging.info("Data loaded successfully:")
        print(data.head())
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    # Check TensorFlow GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logging.info("TensorFlow GPU Availability:")
    print(gpus)

    # Check DummyVecEnv import and initialization
    try:
        # Initialize a minimal environment for testing
        test_env = StockTradingEnv(data)
        env = DummyVecEnv([lambda: test_env])
        logging.info("DummyVecEnv imported and initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing DummyVecEnv: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python verify_setup.py <path_to_csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)

