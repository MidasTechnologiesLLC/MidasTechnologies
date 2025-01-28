Below is a **step-by-step** explanation of how each script (`main.py` for training and `use_dqn.py` for inference) works. We will trace **what** each function does, **how** data flows from one part to the next, and **why** certain design choices (like the advanced indicators and reward function) appear.

---
## **A. `main.py`** (Model Training + Reinforcement Learning)

### **1. Imports and Setup**

1. **Imports**:  
   - `numpy`, `pandas`, etc. handle data.  
   - `tensorflow.keras` for building/training the LSTM.  
   - `optuna` for hyperparameter tuning.  
   - `gym` + `stable_baselines3` for RL environment and DQN.  
   - Some standard Python libraries (`argparse`, `logging`, `sys`).

2. **`os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`**: Lowers TensorFlow logging so we don’t get cluttered with debug messages.  
3. **`logging.basicConfig(...)`**: Configures the logging format and level.

### **2. `load_data(file_path)`**

- Logs a message “Loading data from: ...”  
- Tries to **read a CSV** with a `time` column (parsed as dates).  
- Renames columns: `time → Date`, `open → Open`, etc., to unify naming.  
- Sorts by `Date` and resets index.  
- Logs the final column list and returns a **pandas DataFrame**.

### **3. Technical Indicators** (functions)

- **`compute_rsi()`:**  
  - Takes price series → calculates daily difference → rolling average of gains vs. losses → returns RSI values.  
- **`compute_macd()`:**  
  - Uses exponential moving averages → returns MACD histogram (`macd_line - signal_line`).  
- **`compute_obv()`:**  
  - Sums or subtracts volume based on sign of price changes to get On-Balance Volume.  
- **`compute_adx()`:**  
  - A “pseudo-ADX” using True Range / Close.  
- **`compute_bollinger_bands()`:**  
  - Rolling SMA ± N * rolling std → plus bandwidth.  
- **`compute_mfi()`:**  
  - Money Flow Index: uses typical price, volume, direction.  
- **`calculate_technical_indicators(df)`:**  
  - Logs “Calculating technical indicators...”  
  - Calls the above indicator computations for each row.  
  - Stores them as new columns in `df`: `RSI`, `MACD`, `OBV`, `ADX`, `BB_Upper`, `BB_Lower`, `BB_Width`, `MFI`, plus some moving averages.  
  - Drops rows with `NaN`.  
  - Logs success, returns the updated `df`.

### **4. `parse_arguments()`**

- Uses `argparse` so we can do `python main.py my_data.csv`.  
- Returns parsed arguments with `args.csv_path`.

### **5. `main()` Function**

1. **Argument Parsing**:  
   - `args = parse_arguments()` → obtains `csv_path`.  

2. **Load & Process Data**:
   - `data = load_data(csv_path)` → the CSV is read.  
   - `data = calculate_technical_indicators(data)` → advanced indicators are computed.

3. **Build Feature Set**:
   - A list of `feature_columns` is created, deliberately **excluding** `'Close'`.  
   - We keep `'Close'` as the `target_column`.  
   - The script reorders `data` to `[Date] + feature_columns + [Close]` and drops leftover `NaN`.  

4. **Scaling**:
   - `scaler_features` for `feature_columns`, `scaler_target` for `Close`.  
   - Fit them on the entire dataset and transform to get `X_scaled` and `y_scaled`.  

5. **Sequence Creation** (`create_sequences`):
   - The LSTM model wants sliding-window sequences.  
   - A `window_size = 15`.  
   - For each row in scaled data, it grabs the previous 15 rows as input and the 16th as the target.  
   - Returns arrays `X, y`.

6. **Train/Val/Test Split**:
   - 70% for training, 15% for validation, 15% for test.  
   - Logs shapes of `X_train`, etc.

7. **`configure_device()`**:
   - Checks for GPU presence with `tf.config.list_physical_devices('GPU')`.  
   - If GPU found, sets memory growth. Otherwise logs “No GPU detected, using CPU.”

8. **Build LSTM**: `build_advanced_lstm(input_shape, hyperparams)`
   - Creates a `Sequential` model with `num_lstm_layers` BiLSTM layers.  
   - Each layer has `lstm_units`, a dropout, possibly `return_sequences=True`.  
   - A final Dense(1) output for regression.  
   - The chosen optimizer is either `Adam` or `Nadam`, with a given learning rate, `decay`, etc.  
   - Compiles with `Huber` loss and `mae` metric.

9. **Optuna Tuning**:
   1. `objective(trial)`:  
      - Suggest hyperparameters: number of LSTM layers, units, dropout, learning rate, optimizer, decay.  
      - Build a model with those hyperparams.  
      - Fit on `(X_train, y_train)` with early stopping + reduce LR, returning the best validation MAE.  
      - The pruning callback can prune bad trials.  
   2. `study = optuna.create_study(...)` → optimize with `n_trials=50`.  
   3. `best_params = study.best_params` → logs them.

10. **Train Best Model**:
    - Rebuild the LSTM with the best params from Optuna.  
    - Fit for up to 300 epochs with early stopping + reduce LR.  
    - You see epoch logs (`Epoch 1/300 ...`).

11. **Evaluate**: `evaluate_model(best_model, X_test, y_test)`
    - Gets predictions from the best model on `X_test`.  
    - Clamps them to `[0,1]`, then inverse transforms with `scaler_target`.  
    - Logs MSE, RMSE, MAE, R², and directional accuracy.  
    - Plots Actual vs. Predicted and saves `actual_vs_predicted.png`.  
    - Prints first 40 predictions as a tabulated list.

12. **Save**:
    - `best_model.save("optimized_lstm_model.h5")`.  
    - `joblib.dump(...)` scalers.  
    - Logs success.

13. **Reinforcement Learning Environment**:
    - Defines `class StockTradingEnv(gym.Env):` which uses the same **`feature_columns`**.  
    - `__init__` sets `initial_balance`, `balance`, `shares_held`, etc.  
    - `_next_observation()` returns normalized features + `[balance/initial_balance, shares_held/100, cost_basis/initial_balance]`.  
    - `step(action)`: if action=2 → buy, 0 → sell, 1 → hold. Recalculates net worth.  
    - Reward is `(net_worth - initial_balance)` each step.  
    - Logs environment steps in `render`.

14. **`train_dqn_agent(env)`**:
    - Builds a DQN with `MlpPolicy`, `learning_rate=1e-3`, `buffer_size=10000`, etc.  
    - `model.learn(total_timesteps=100000)` → trains on the environment.  
    - Saves `dqn_stock_trading.zip`.  

15. **Creating and Training the DQN**:
    - `trading_env = StockTradingEnv(data)` → wrapped in `DummyVecEnv`.  
    - `dqn_model = train_dqn_agent(trading_env)`.

16. **Logs** “All tasks complete. Exiting.” and ends.

---

## **B. `use_dqn.py`** (Inference on the Trained DQN)

### **1. Imports + Setup**

- Same logic for standard libraries plus `stable_baselines3` and a few helper functions for indicators.  
- `StockTradingEnv` class is defined again, but with the **same** columns as in `main.py`’s RL environment.

### **2. `compute_technical_indicators(df)`**  
- **Must** replicate the same advanced columns so that the environment’s state matches the one used during training.  
- Creates `RSI`, `MACD`, `OBV`, `ADX`, `BB_Upper`, `BB_Lower`, `BB_Width`, `MFI`, and so on.

### **3. `StockTradingEnv`**  
- This is the environment used in inference.  
- Notice it has `feature_columns` = 17 advanced columns + 3 more fields in the observation → total shape **(20,)**.  
- That matches the environment used in `main.py` so `DQN.load(...)` will not complain about shape mismatch.

### **4. `parse_arguments()`**  
- Allows a `-s` or `--show-steps` argument to specify how many final steps to display (default 15, max 300).

### **5. `main()`**:

1. **Arg Parsing**:  
   - `args = parse_arguments()` → `steps_to_display = min(args.show_steps, 300)`.

2. **Load CSV**:
   - Reads `BAT.csv` → renames columns, sorts by date.

3. **Compute Indicators**:
   - The same `compute_technical_indicators(df)` to ensure consistent columns.

4. **Create Environment**:
   - `raw_env = StockTradingEnv(df)`.  
   - `vec_env = DummyVecEnv([lambda: raw_env])`.

5. **Load DQN**:
   - `model = DQN.load("dqn_stock_trading.zip", env=vec_env)`.  
   - Must match the environment’s observation space shape (20,).

6. **Run Inference**:
   - `obs = vec_env.reset()`, `done = [False]`.  
   - While not done, do `action, _ = model.predict(obs, deterministic=True)` → step.  
   - Keep track of total reward, step data (balance, shares, net worth each step).

7. **Final Summary**:
   - Prints final net worth, final profit, sum of rewards.  
   - Counts how many buy/sell/hold actions.  
   - Displays the last `steps_to_display` steps in a table.

8. **Exits**.

---

## **Flow of Data in `main.py`**

1. **User runs**: `python main.py your_data.csv`.  
2. The script loads CSV → advanced indicators → scales → sequences → LSTM model.  
3. **Optuna** tries different hyperparams. For each trial, a partial model is trained for 100 epochs or so. It picks the best.  
4. The final LSTM is trained with those best hyperparams for up to 300 epochs.  
5. Evaluate on the test set → logs metrics → prints a table.  
6. The script builds an RL environment, **train_dqn_agent** for 100k steps. That environment references the same columns but ignoring the close for the observation, creating a shape (20,).  
7. Saves the DQN model.  
8. Done.

---

## **Flow of Data in `use_dqn.py`**

1. **User runs**: `python use_dqn.py`.  
2. Loads the same CSV (`BAT.csv`) → advanced indicators so columns match.  
3. **StockTradingEnv** is created with 17 columns → total obs shape (20,).  
4. `DQN.load(...)` loads `dqn_stock_trading.zip`.  
5. The agent is run step by step until done, collecting actions, rewards, balances, etc.  
6. Prints final stats, final steps, net worth, etc.

---

## **Summary**

- **`main.py`** is a multi-phase script:
  1. **Data ingestion + indicator generation**.  
  2. **Sequence creation** for LSTM.  
  3. **Optuna hyperparameter search** for best LSTM config.  
  4. **Final LSTM training** with the chosen hyperparams.  
  5. **Evaluation** (plots, table).  
  6. **RL environment** creation → **DQN** training.  
  7. Saves the final DQN model.

- **`use_dqn.py`** re-creates the same environment with the same features to **load** the trained DQN and run inference, printing the agent’s final performance.

The main interplay is that both scripts share the same advanced feature list so that the observation space is consistent for the RL environment, ensuring that the **DQN** model can be loaded without shape mismatch. The LSTM portion is separate but also uses the advanced indicators and excludes `Close` from inputs—**only** using it as a label for supervised learning.
