import argparse
import gym
import numpy as np
import pandas as pd
from tabulate import tabulate

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


###############################
# 1. HELPER FUNCTIONS
###############################
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=span_signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram

def compute_adx(df, window=14):
    # Placeholder for ADX calculation
    return df['Close'].rolling(window=window).std()

def compute_obv(df):
    # On-Balance Volume calculation
    OBV = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return OBV

def compute_technical_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['STDDEV_5'] = df['Close'].rolling(window=5).std()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = compute_macd(df['Close'])
    df['ADX'] = compute_adx(df)
    df['OBV'] = compute_obv(df)
    df.dropna(inplace=True)
    return df


###############################
# 2. ENVIRONMENT DEFINITION
###############################
class StockTradingEnv(gym.Env):
    """
    Simple environment using older Gym API for SB3.
    """
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_steps = len(df)
        self.current_step = 0
        self.shares_held = 0
        self.cost_basis = 0

        self.feature_columns = [
            'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'STDDEV_5',
            'RSI', 'MACD', 'ADX', 'OBV', 'Volume',
            'Open', 'High', 'Low'
        ]

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
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
        return self._next_observation()

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        # BUY
        if action == 2:
            total_possible = self.balance // current_price
            shares_bought = int(total_possible)
            if shares_bought > 0:
                prev_shares = self.shares_held
                self.balance -= shares_bought * current_price
                self.shares_held += shares_bought
                self.cost_basis = (
                    (self.cost_basis * prev_shares) + (shares_bought * current_price)
                ) / self.shares_held

        # SELL
        elif action == 0:
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
                self.cost_basis = 0

        self.net_worth = self.balance + self.shares_held * current_price
        self.current_step += 1
        done = (self.current_step >= self.max_steps - 1)

        reward = self.net_worth - self.initial_balance
        obs = self._next_observation()
        return obs, reward, done, {}

    def _next_observation(self):
        row = self.df.loc[self.current_step, self.feature_columns].values
        max_val = np.max(row) if np.max(row) != 0 else 1.0
        row_norm = row / max_val

        additional = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100.0,
            self.cost_basis / self.initial_balance
        ], dtype=np.float32)

        obs = np.concatenate([row_norm, additional]).astype(np.float32)
        return obs

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
    parser.add_argument("-s", "--show-steps", type=int, default=15,
                        help="Number of final steps to display in the summary (default: 15, max: 300).")
    return parser.parse_args()


###############################
# 4. MAIN FUNCTION
###############################
def main():
    args = parse_arguments()
    # Bound how many steps we show at the end
    steps_to_display = min(args.show_steps, 300)

    # 1) Load CSV
    df = pd.read_csv('BAT.csv')
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
    df = compute_technical_indicators(df)
    if 'volume' in df.columns and 'Volume' not in df.columns:
        df.rename(columns={'volume': 'Volume'}, inplace=True)

    # 2) Instantiate environment
    raw_env = StockTradingEnv(df)
    vec_env = DummyVecEnv([lambda: raw_env])

    # 3) Load your DQN model
    model = DQN.load("dqn_stock_trading.zip", env=vec_env)

    # 4) Run inference
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

    # 5) Print final summary
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

    # 6) Show the last N steps, where N=steps_to_display
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


if __name__ == "__main__":
    main()

