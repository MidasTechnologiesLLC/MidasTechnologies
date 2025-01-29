Below is a **big-picture explanation** of how the code is structured and *why* you have both **LSTM** (with Optuna hyperparameter tuning) and **DQN** (for reinforcement learning). It also clarifies how each component is supposed to work, what is happening in the environment, and why the agent’s final performance might look confusing if you expect it to strictly follow the LSTM predictions.

---

## 1. **Two Separate Approaches**: **(A) LSTM** vs. **(B) DQN**

- **(A) LSTM (Supervised Learning)**  
  - Goal: Predict future close prices from historical indicators.  
  - You do **Optuna** hyperparameter search to find the best LSTM configuration (number of layers, units, dropout, etc.).  
  - You end up with an LSTM model that, for any given window of past data, outputs a predicted close price.

- **(B) DQN (Reinforcement Learning)**  
  - Goal: Learn a *trading policy* (when to buy, sell, or hold) to maximize cumulative profit.  
  - The RL environment steps day-by-day (or bar-by-bar) through the historical dataset. The agent sees “State” (technical indicators + account info), chooses an action, gets a reward (net worth change).  
  - Over many episodes/timesteps, the agent tries to find the best sequence of buy/sell/hold actions that yields the highest total reward.

**These are two distinct paradigms**:  
1. **LSTM** tries to *accurately forecast* future prices.  
2. **DQN** tries to *learn a trading strategy* that yields high profit—**even if** the underlying predictions are not explicitly used.

---

## 2. **How They Typically Interact** (If You Combine Them)

In the scripts you’ve seen, **the code doesn’t actually pass LSTM predictions into the DQN**. Instead, the DQN environment simply sees the same features (RSI, MACD, etc.) that the LSTM used, plus account info. The agent then decides actions (buy/sell/hold) purely from that state. The environment calculates a reward (net worth change). Over many steps, the DQN is supposed to figure out how to trade for maximum cumulative profit.

### **Alternative**: Using LSTM *inside* the RL
- One advanced approach is to have the RL environment or agent **use** the LSTM’s predicted next price as part of its observation.  
- Another is to have the LSTM run first, produce a forecast series, and the RL agent trades on that forecast.  
- But the code you have does not appear to do that. It simply trains the LSTM and then trains a DQN environment **separately**.  

Hence, your final code does the following:
1. **Train LSTM** with advanced indicators to see if you get a good predictive model.  
2. **Separately** build a `StockTradingEnv` that steps through the actual historical data row-by-row.  
3. **Train a DQN** on that environment to learn a trading policy from the same indicators.  
4. The DQN’s net profit depends on how well it learns to interpret the indicators to buy low, sell high, or hold.

---

## 3. **What the DQN Actually Does** in This Code

1. **Initialization**  
   - The environment starts with `balance = 10000`, `shares_held = 0`.  
   - The agent sees a state: (SMA_5, RSI, etc., plus balance ratio, shares ratio, cost basis ratio).  
2. **Step Through Each Day**  
   - Action = 0 (Sell), 1 (Hold), 2 (Buy).  
   - If *Buy*: it invests as much as possible based on the current balance.  
   - If *Sell*: it closes the position, returning to all-cash.  
   - The environment updates net worth, increments the current step, calculates reward.  
3. **Reward**  
   - The code sets `reward = net_worth - initial_balance` on each step or (in some versions) `reward = net_worth - prev_net_worth`.  
   - The agent tries to **maximize** the sum of these step-based rewards over the entire episode.  

**In theory**, the agent should learn how to buy before price rises and sell before price falls, maximizing net worth by the final step. However:

- **If the environment’s reward definition is “net_worth - initial_balance”** on each step, the sum of these partial rewards can be misleading. The final net worth might return to the initial balance while partial step rewards sum to a negative number. This is a quirk of how the incremental reward is defined.  
- **If the environment ends with net worth = 10k** (the same as initial), it means the agent didn’t manage to produce a final profit in that particular run, even though it took many buy/sell steps.  

---

## 4. **Why There’s No Direct Use of the LSTM Inside the DQN**

- The code doesn’t show anything like “use the best_model from LSTM to get predictions” inside the RL environment.  
- Instead, the RL environment is purely an **indicator-based** approach. It sees (SMA, RSI, etc.) for each day, decides an action.  
- This is a typical architecture in some code bases: they first do an LSTM or XGBoost model to forecast prices *on the side*, and they also do an RL approach. They’re parallel attempts at a solution, not necessarily integrated.

If your goal is:  
> “**Use LSTM’s predicted next day close** as an observation for the DQN agent, so it can trade based on that forecast,”  
then you’d explicitly pass `lstm_model.predict(current_window_data)` into the environment’s observation. That is not done in the current code, so you see them as separate modules.

---

## 5. **Yes, the DQN Simulates a Trading Environment** Over a Year

> *“It should try to maximize its profit on a day to day basis for some amount of days (ideally a full year).”*

That is precisely the idea. The environment loops day by day over the historical dataset. At each day:
- The agent sees indicators + account state.  
- The agent picks buy/sell/hold.  
- The environment updates net worth, calculates a reward, goes to the next day.  

At the end, you see how much total net worth is left. If the agent is good, it has more than $10k; if not, it’s at or below $10k.

**In your result**: it ended up at $10k final net worth → zero final profit, meaning the learned policy is effectively break-even or not very successful. Possibly more training or different hyperparameters for the RL might help, or a different reward definition.

---

## 6. **Summary / Next Steps**

1. **The LSTM** + **Optuna** finds an accurate or best-fitting model for price prediction, but the code doesn’t feed those predictions to the DQN.  
2. **The DQN** environment tries to buy/sell/hold purely from the same raw indicators. If it ends with net worth = $10k, it means no net profit in that historical backtest.  
3. If you **want** the DQN to make use of the LSTM predictions, you’d add code in `_get_obs()` or the environment to:  
   - Compute or retrieve the LSTM’s predicted next close.  
   - Include that as part of the observation.  
4. If you want a simpler step-based reward that sums more intuitively to “final profit,” define `reward = net_worth - prev_net_worth` each step. Then the sum of step rewards will match `(final_net_worth - initial_balance)`.

---

### **Bottom Line**

- The DQN is indeed simulating day-to-day trading with $10k starting capital, but it’s not actively using the LSTM forecasts in the environment as currently coded.  
- It tries to maximize day-to-day profit via an RL approach on the same historical data.  
- If you see zero final profit, it means the agent’s policy ended up break-even on that historical run. Over many runs or better hyperparameters, it might do better. Or you can integrate the LSTM predictions into the RL state to potentially improve performance.
