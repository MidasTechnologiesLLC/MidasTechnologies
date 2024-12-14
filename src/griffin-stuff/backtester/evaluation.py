import numpy as np

def evaluate_indicator_accuracy(df, price_col="Close", horizon=1):
    """
    Evaluate how often indicator signals predict the correct next-day price direction.
    
    Logic:
    - If signal[i] = 1 (bullish), correct if price[i+horizon] > price[i].
    - If signal[i] = -1 (bearish), correct if price[i+horizon] < price[i].
    - If signal[i] = 0, skip.
    """
    correct = 0
    total = 0
    
    for i in range(len(df) - horizon):
        sig = df['signal'].iloc[i]
        if sig == 0:
            continue
        future_price = df[price_col].iloc[i + horizon]
        current_price = df[price_col].iloc[i]
        
        if sig == 1 and future_price > current_price:
            correct += 1
        elif sig == -1 and future_price < current_price:
            correct += 1
        
        if sig != 0:
            total += 1
    
    if total == 0:
        return np.nan  # No signals generated
    return correct / total
