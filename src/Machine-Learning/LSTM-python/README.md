# I got Bitched. Fuck Python
# Below is a breakdown of how **`main.py`** works, what files it generates, and how all the pieces (LSTM model training, Optuna tuning, DQN trading agent, etc.) fit together. I’ll also cover the CSV structure, how to interpret metrics, how to handle the various files, and how to potentially adapt this code for real-time predictions or a live trading bot.

---

## 1. **High-Level Flow of `main.py`**

1. **Imports & Logging Setup**  
   - Imports all necessary libraries (numpy, pandas, sklearn, TensorFlow, XGBoost, Optuna, Stable Baselines, etc.)  
   - Sets up basic logging with timestamps and message levels.
   
2. **Argument Parsing**  
   - Uses `argparse` to expect one argument: the path to the CSV file.  
   - You run the script like `python main.py your_data.csv`.

3. **Data Loading & Preprocessing**  
   - **`load_data(file_path)`**: Reads the CSV with pandas, renames columns (e.g., `time` → `Date`, `open` → `Open`, etc.), sorts by date, and returns a cleaned DataFrame.  
   - **`calculate_technical_indicators(df)`**:  
     - Computes various indicators (SMA, EMA, RSI, MACD, ADX, OBV).  
     - Drops rows with `NaN` values (because rolling windows can produce NaNs).  
   - After these steps, the data is ready for feature selection.

4. **Feature Selection & Scaling**  
   - Chooses certain columns as features (`feature_columns`) plus the target (`Close`).  
   - Scales features and target with `MinMaxScaler`, which normalizes values to [0, 1].  
   - Prepares sequences of length `window_size` (default = 15) for LSTM training via **`create_sequences()`**.

5. **Train/Validation/Test Split**  
   - Splits sequences into 70% for training, 15% for validation, and 15% for testing.  
   - This yields `X_train`, `X_val`, `X_test` and corresponding `y_train`, `y_val`, `y_test`.

6. **Device Configuration**  
   - Checks for GPUs and configures TensorFlow to allow memory growth on the available GPU(s).

7. **Model Building and Hyperparameter Tuning**  
   - **`build_advanced_lstm(...)`**: Creates a multi-layer, bidirectional LSTM with optional dropout, user-defined optimizer, learning rate, etc.  
   - **Optuna**:  
     - The `objective(trial)` function defines which hyperparameters to search.  
     - It trains an LSTM for each set of hyperparameters.  
     - Minimizes the validation MAE to find the best combination.  
     - Runs `study.optimize(objective, n_trials=50)` to try up to 50 hyperparameter sets.  
   - The best hyperparameters are then retrieved (`study.best_params`).

8. **Train the Best LSTM Model**  
   - Re-builds the LSTM with the best hyperparameters found.  
   - Uses callbacks (`EarlyStopping`, `ReduceLROnPlateau`) for better generalization.  
   - Trains up to 300 epochs or until early stopping.

9. **Model Evaluation**  
   - **`evaluate_model(...)`**:  
     - Gets predictions (`model.predict(X_test)`).  
     - Inverse-transforms them to the original scale (because we had scaled them).  
     - Computes **MSE**, **RMSE**, **MAE**, **R2**, and **directional accuracy**.  
     - Saves a plot called **`actual_vs_predicted.png`** comparing actual and predicted test prices.  
     - Prints the first 40 predictions in a tabular format.

10. **Save Model & Scalers**  
    - Saves the Keras model to **`optimized_lstm_model.h5`**.  
      - _(Keras warns this is a legacy format and suggests using `.keras` file extension—more on that later.)_  
    - Saves the scalers to **`scaler_features.save`** and **`scaler_target.save`** using `joblib`.

11. **Reinforcement Learning (DQN) Setup**  
    - Defines a custom Gym environment **`StockTradingEnv`** that simulates a simple stock trading scenario:  
      - Discrete action space: **0 = Sell**, **1 = Hold**, **2 = Buy**.  
      - Observations: scaled features + current balance + shares held + cost basis.  
      - Reward is the change in net worth.  
    - Uses **Stable Baselines 3** (`DQN`) to train an agent in that environment for 100,000 timesteps.  
    - Saves the agent as **`dqn_stock_trading.zip`**.

12. **Finally**:  
    - The script ends after the RL agent training completes.

---

## 2. **Explanation of Generated Files**

After you run `main.py`, you typically end up with these files:

1. **`optimized_lstm_model.h5`**  
   - The final trained LSTM model, saved in the HDF5 format.  
   - **Keras** now recommends using `model.save('optimized_lstm_model.keras')` or `keras.saving.save_model(model, 'optimized_lstm_model.keras')` for a more modern format.

2. **`scaler_features.save`**  
   - A `joblib` file containing the fitted `MinMaxScaler` for the features.

3. **`scaler_target.save`**  
   - Another `joblib` file for the target variable’s `MinMaxScaler`.

4. **`actual_vs_predicted.png`**  
   - A PNG plot comparing the actual vs. predicted close prices from the test set.

5. **`dqn_stock_trading.zip`**  
   - The trained DQN agent from Stable Baselines 3.

6. **`dqn_stock_tensorboard/`**  
   - A directory containing TensorBoard logs for the DQN training process.  
   - You can inspect these logs by running `tensorboard --logdir=./dqn_stock_tensorboard`.

7. **Other legacy or auxiliary files** you may have in the same folder:  
   - **`enhanced_lstm_model.h5`**, **`prediction_vs_actual.png`**, **`policy.pth`**, **`_stable_baselines3_version`**, etc., come from either old runs or intermediate attempts. You can clean them up if you no longer need them.

---

## 3. **Your CSV (`time,open,high,low,close,Volume`)**

An example snippet:
```
time,open,high,low,close,Volume
2024-01-08T09:30:00-05:00,59.23,59.69,59.03,59.53,4335
...
```
- The script renames these columns to `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.  
- It sorts by `Date` and starts computing features.

Because the script expects `Date` sorted in ascending order, make sure all timestamps follow the same format.

---

## 4. **Interpreting the Evaluation Metrics**

When the script prints:

- **MSE (Mean Squared Error)**  
- **RMSE (Root MSE)**  
- **MAE (Mean Absolute Error)**  
- **R² (Coefficient of Determination)**  
- **Directional Accuracy**  

### Example Output Interpretation

- **MSE: 0.0838** → The average of squared errors on the original (inversely scaled) price scale is relatively low.  
- **RMSE: 0.2895** → The square root of that MSE, so on average the model’s predictions deviate by about 0.29 from the actual close price.  
- **MAE: 0.1836** → On average, the absolute deviation is ~0.18.  
- **R²: 0.993** → Very high, suggests the model explains 99.3% of the variance in price.  

However, a **directional accuracy** of ~0.48 suggests the model is not great at predicting whether the price goes up or down from one timestep to the next. It’s close to random guessing (50%). This can happen if the model is good at capturing overall magnitude but not short-term direction.

If you need the model to be directionally correct more often (for trading), consider:
- Shifting the target to be the price change or return (rather than the absolute price).  
- Using classification-based approach (up/down) or building a custom loss function that focuses more on directional accuracy.

---

## 5. **How to Improve or Change the Metric Outputs**

1. **Custom Metrics**:  
   - You can add them to `model.compile(metrics=[...])` if they’re supported by Keras.  
   - Or you can compute them manually in the `evaluate_model` function (like you already do for R², directional accuracy, etc.).

2. **Reducing Warnings**:  
   - **HDF5 warning**: Instead of `best_model.save('optimized_lstm_model.h5')`, do:
     ```python
     best_model.save('optimized_lstm_model.keras')
     ```
     Or:
     ```python
     keras.saving.save_model(best_model, 'optimized_lstm_model.keras')
     ```
   - **Gym vs. Gymnasium warning** in Stable Baselines:  
     - You can switch to Gymnasium by installing `gymnasium` and adapting the environment accordingly:
       ```python
       import gymnasium as gym
       ```
       Then use `gymnasium.make(...)`.  
       - But as long as it’s working, the warning is mostly informational.

3. **Remove Unused Files**:  
   - If certain files are no longer used or were generated by old runs, just delete them to keep your workspace clean.

---

## 6. **Using the DQN Agent**

### How It’s Being Trained
- **`StockTradingEnv`** is a simplified environment that steps through your historical data row by row (`self.max_steps = len(df)`).  
- Each step, you pick an action (Sell, Hold, or Buy).  
- The environment updates your balance, shares held, cost basis, and net worth accordingly.  
- The reward is `(net_worth - initial_balance)`, i.e. how much you’ve gained or lost.

### How to Deploy It
1. **After Training**: You have **`dqn_stock_trading.zip`** saved.  
2. **Load the Model** in a separate script or Jupyter notebook:
   ```python
   from stable_baselines3 import DQN
   from stable_baselines3.common.vec_env import DummyVecEnv

   # Recreate the same environment
   env = StockTradingEnv(your_dataframe)
   env = DummyVecEnv([lambda: env])

   # Load the trained agent
   model = DQN.load("dqn_stock_trading.zip", env=env)
   ```
3. **Run Predictions**:
   ```python
   obs = env.reset()
   done = False
   while not done:
       # Model predicts the best action
       action, _states = model.predict(obs, deterministic=True)
       obs, reward, done, info = env.step(action)
       env.render()
   ```
   This will step through the environment again, but now with your trained agent. In a real-time scenario, you’d need a streaming environment that updates with new data in small increments (e.g., each new minute’s bar).

---

## 7. **Transition to Real-Time (“Live”) Predictions**

1. **Live Price Feed**:  
   - You would replace the static CSV with a real-time feed (e.g., an API from a broker or a data provider).  
   - Keep a rolling window of the last `window_size` data points, compute your indicators on the fly.

2. **Online or Incremental Updates**:  
   - For an LSTM, you typically retrain or fine-tune it with new data over time. Or you load the existing model and just do forward passes for the new window.  
   - The code that constructs sequences would run each time you get a new data point, but typically you’d keep a queue or buffer of the recent `N` bars.

3. **Deploying the DQN**:
   - Similarly, in a real environment, each new bar triggers `env.step(action)`. The environment’s “current step” is the latest bar.  
   - You might have to rewrite the environment’s logic so it only advances by one bar at a time in real-time, rather than iterating over the entire historical dataset.

---

## 8. **Summary**

- **`main.py`** orchestrates:
  1. Data Loading + Preprocessing  
  2. Feature Engineering (SMA, EMA, RSI, MACD, ADX, OBV)  
  3. LSTM Hyperparameter Tuning with Optuna  
  4. Best Model Training + Saving + Evaluation  
  5. Simple RL Environment + DQN Training + Saving

- **Key Files** Generated:  
  - `optimized_lstm_model.h5` (or `.keras`) → your final Keras LSTM model.  
  - `scaler_features.save`, `scaler_target.save` → joblib-saved scalers.  
  - `actual_vs_predicted.png` → visual of test set predictions.  
  - `dqn_stock_trading.zip` → trained RL agent.  
  - `dqn_stock_tensorboard/` → logs for the RL training.

- **Interpreting Metrics**:  
  - High R² with lower directional accuracy implies it fits magnitudes well but struggles with sign changes.  
  - Potential improvement: feature engineering for short-term direction or a classification approach for up vs. down.

- **Using the DQN Agent**:  
  - `StockTradingEnv` is a toy environment stepping over historical data.  
  - Real-time adaptation requires modifying how the environment receives data.

- **Warnings**:  
  - Switch `.h5` → `.keras` to remove the Keras format warning.  
  - Possibly switch from Gym to Gymnasium to remove the stable-baselines3 compatibility warning.

---

### Next Steps / Tips

1. **Clean Up Legacy Files**: If you have old models or references (like `enhanced_lstm_model.h5`), remove or rename them.  
2. **Custom Loss / Custom Metrics**: If you want to focus on direction, consider a custom loss function or a classification-based approach.  
3. **Try Different RL Algorithms**: DQN is just one method. PPO, A2C, or SAC might handle continuous or more complex action spaces.  
4. **Hyperparameter Range**: Expand or refine your Optuna search space. For instance, trying different `window_sizes` or different dropout regularization strategies.  
5. **Feature Engineering**: More sophisticated indicators or external features (e.g., news sentiment, fundamental data) might help.

All in all, your script is already quite comprehensive. You have an advanced LSTM pipeline for regression plus a DQN pipeline for RL. The main things to refine will be:
- **Data quality**  
- **Indicator relevance**  
- **Directional vs. magnitude accuracy**  
- **Live streaming vs. historical backtesting**  

Once you address those, your system will be closer to a real-time AI/bot capable of forecasting or trading on new data.
