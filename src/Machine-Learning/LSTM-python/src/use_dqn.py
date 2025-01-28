import os
import sys
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

###############################
# 1. HELPER FUNCTIONS
###############################
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / (loss + 1e-9)
    return 100 - (100 / (1 + RS))

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line

def compute_obv(df):
    signed_volume = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    return signed_volume.cumsum()

def compute_adx(df, window=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    tr = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    adx_placeholder = tr.rolling(window=window).mean() / (df['Close'] + 1e-9)
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

def compute_technical_indicators(df):
    """
    Same advanced indicators as used in main.py:
    RSI, MACD, OBV, ADX, Bollinger, MFI, plus rolling means & std.
    """
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
    return df


###############################
# 2. Enhanced Environment
###############################
class StockTradingEnv(gym.Env):
    """
    Environment matching the advanced environment from main.py:
      - 17 columns
      - incremental reward
      - transaction cost
      - fundamental checks (RSI>70 => skip buy)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.transaction_cost = transaction_cost

        self.max_steps = len(df)
        self.current_step = 0
        self.shares_held = 0
        self.cost_basis = 0

        # Must EXACTLY MATCH your main.py 'feature_columns'
        self.feature_columns = [
            'SMA_5','SMA_10','EMA_5','EMA_10','STDDEV_5',
            'RSI','MACD','ADX','OBV','Volume','Open','High','Low',
            'BB_Upper','BB_Lower','BB_Width','MFI'
        ]
        # 17 columns + 3 add'l = 20
        self.action_space = spaces.Discrete(3)  # 0=Sell,1=Hold,2=Buy
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(len(self.feature_columns)+3,),
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
        row = self.df.iloc[self.current_step]
        row_feats = row[self.feature_columns]
        max_val = row_feats.max() if row_feats.max()!=0 else 1.0
        row_norm = row_feats / max_val

        additional = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100.0,
            self.cost_basis / (self.initial_balance + 1e-9)
        ], dtype=np.float32)

        return np.concatenate([row_norm.values, additional]).astype(np.float32)

    def step(self, action):
        prev_net_worth = self.net_worth

        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        rsi_value = row['RSI']

        # optional fundamental check => skip buy if RSI>70
        can_buy = (rsi_value < 70)

        if action == 2 and can_buy:  # BUY
            shares_bought = int(self.balance // current_price)
            if shares_bought > 0:
                cost = shares_bought * current_price
                fee = cost * self.transaction_cost
                self.balance -= (cost + fee)
                prev_shares = self.shares_held
                self.shares_held += shares_bought
                # Weighted average cost
                self.cost_basis = (
                    (self.cost_basis * prev_shares) + (shares_bought * current_price)
                ) / self.shares_held

        elif action == 0:  # SELL
            if self.shares_held > 0:
                revenue = self.shares_held * current_price
                fee = revenue * self.transaction_cost
                self.balance += (revenue - fee)
                self.shares_held = 0
                self.cost_basis = 0

        # recalc net worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.current_step += 1
        done = (self.current_step >= self.max_steps - 1)

        # incremental reward
        reward = self.net_worth - prev_net_worth

        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step} | "
              f"Balance: {self.balance:.2f} | "
              f"Shares: {self.shares_held} | "
              f"NetWorth: {self.net_worth:.2f} | "
              f"Profit: {profit:.2f}")


###############################
# 3. ARGUMENT PARSING
###############################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Use a trained DQN model to run a stock trading simulation.")
    parser.add_argument("--csv", type=str, default="BAT.csv",
                        help="Path to the CSV data (same format as main.py). Default: 'BAT.csv'")
    parser.add_argument("-s", "--show-steps", type=int, default=15,
                        help="Number of final steps to display in the summary (default: 15, max: 300).")
    return parser.parse_args()


###############################
# 4. MAIN FUNCTION
###############################
def main():
    args = parse_arguments()
    steps_to_display = min(args.show_steps, 300)
    csv_path = args.csv

    try:
        # 1) Load CSV
        df = pd.read_csv(csv_path)
        rename_mapping = {
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        }
        df.rename(columns=rename_mapping, inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 2) Compute advanced indicators
        df = compute_technical_indicators(df)
        if 'volume' in df.columns and 'Volume' not in df.columns:
            df.rename(columns={'volume': 'Volume'}, inplace=True)

        # 3) Create the environment
        raw_env = StockTradingEnv(df, initial_balance=10000, transaction_cost=0.001)
        vec_env = DummyVecEnv([lambda: raw_env])

        # 4) Load your DQN model
        model = DQN.load("dqn_stock_trading.zip", env=vec_env)

        # 5) Run inference
        obs = vec_env.reset()
        done = [False]
        total_reward = 0.0
        step_data = []
        step_count = 0

        underlying_env = vec_env.envs[0]

        while not done[0]:
            step_count += 1
            action, _ = model.predict(obs, deterministic=True)
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

        # 6) Print final summary
        print("\n=== DQN Agent Finished ===")
        print(f"Total Steps Taken: {step_count}")
        print(f"Final Net Worth: {final_net_worth:.2f}")
        print(f"Final Profit: {final_profit:.2f}")
        print(f"Sum of Rewards: {total_reward:.2f}")

        # Count actions
        buy_count = sum(1 for x in step_data if x["Action"] == 2)
        sell_count = sum(1 for x in step_data if x["Action"] == 0)
        hold_count = sum(1 for x in step_data if x["Action"] == 1)
        print(f"Actions Taken -> BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

        # 7) Show the last N steps
        last_n = step_data[-steps_to_display:] if len(step_data) > steps_to_display else step_data
        rows = []
        for d in last_n:
            rows.append([
                d["Step"], d["Action"], f"{d['Reward']:.2f}",
                f"{d['Balance']:.2f}", d["Shares"], f"{d['NetWorth']:.2f}"
            ])
        headers = ["Step", "Action", "Reward", "Balance", "Shares", "NetWorth"]
        print(f"\n== Last {steps_to_display} Steps ==")
        print(tabulate(rows, headers=headers, tablefmt="pretty"))

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected: exiting gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

