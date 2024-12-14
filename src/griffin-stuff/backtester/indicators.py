import pandas as pd
import numpy as np
import ta

def calculate_indicator_signals(df, indicator_name, params, price_col="Close", high_col="High", low_col="Low", volume_col="Volume"):
    """
    Calculates indicator values and generates signals:
    Signal Convention: 1 = Bullish Prediction, -1 = Bearish Prediction, 0 = Neutral
    """
    if price_col not in df.columns:
        raise ValueError(f"{price_col} column not found in the dataframe.")
    
    if indicator_name == "SMA":
        # Trend: price > SMA => bullish, else bearish
        window = params.get("window", 20)
        df['SMA'] = df[price_col].rolling(window).mean()
        df['signal'] = np.where(df[price_col] > df['SMA'], 1, -1)
    
    elif indicator_name == "EMA":
        # Trend: price > EMA => bullish, else bearish
        window = params.get("window", 20)
        df['EMA'] = df[price_col].ewm(span=window, adjust=False).mean()
        df['signal'] = np.where(df[price_col] > df['EMA'], 1, -1)
    
    elif indicator_name == "ADX":
        # Trend: use ADXIndicator
        if high_col not in df.columns or low_col not in df.columns:
            raise ValueError("ADX calculation requires 'High' and 'Low' columns.")
        window = params.get("window", 14)
        adx_indicator = ta.trend.ADXIndicator(high=df[high_col], low=df[low_col], close=df[price_col], window=window)
        df['ADX'] = adx_indicator.adx()
        df['DIP'] = adx_indicator.adx_pos()  # +DI
        df['DIN'] = adx_indicator.adx_neg()  # -DI
        
        # If ADX > 25 and DI+ > DI- => bullish
        # If ADX > 25 and DI- > DI+ => bearish
        # Otherwise => no strong signal
        df['signal'] = 0
        trending_up = (df['DIP'] > df['DIN']) & (df['ADX'] > 25)
        trending_down = (df['DIN'] > df['DIP']) & (df['ADX'] > 25)
        df.loc[trending_up, 'signal'] = 1
        df.loc[trending_down, 'signal'] = -1
    
    elif indicator_name == "RSI":
        # Momentum: RSI > overbought => bearish, RSI < oversold => bullish
        window = params.get("window", 14)
        overbought = params.get("overbought", 70)
        oversold = params.get("oversold", 30)
        df['RSI'] = ta.momentum.rsi(df[price_col], window=window)
        conditions = [
            (df['RSI'] > overbought),
            (df['RSI'] < oversold)
        ]
        values = [-1, 1]
        df['signal'] = np.select(conditions, values, default=0)
    
    elif indicator_name == "MACD":
        # Momentum: MACD line > Signal line => bullish, else bearish
        fastperiod = params.get("fastperiod", 12)
        slowperiod = params.get("slowperiod", 26)
        signalperiod = params.get("signalperiod", 9)
        macd = ta.trend.MACD(df[price_col], window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        df['signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
    
    elif indicator_name == "BollingerBands":
        # Volatility: price near upper band => bearish, near lower band => bullish
        window = params.get("window", 20)
        std_dev = params.get("std_dev", 2)
        bb = ta.volatility.BollingerBands(df[price_col], window=window, window_dev=std_dev)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['signal'] = np.where(df[price_col] >= df['BB_High'], -1,
                        np.where(df[price_col] <= df['BB_Low'], 1, 0))
    
    elif indicator_name == "OBV":
        # Volume: Rising OBV => bullish, falling OBV => bearish
        if volume_col not in df.columns:
            raise ValueError(f"OBV calculation requires '{volume_col}' column.")
        df['OBV'] = ta.volume.on_balance_volume(df[price_col], df[volume_col])
        df['OBV_Change'] = df['OBV'].diff()
        df['signal'] = np.where(df['OBV_Change'] > 0, 1, np.where(df['OBV_Change'] < 0, -1, 0))
    
    elif indicator_name == "MeanReversionSignal":
        # Mean Reversion: price > mean => bearish, price < mean => bullish
        window = params.get("window", 10)
        df['mean'] = df[price_col].rolling(window).mean()
        df['signal'] = np.where(df[price_col] > df['mean'], -1, 
                        np.where(df[price_col] < df['mean'], 1, 0))
    
    else:
        raise ValueError(f"Unknown indicator: {indicator_name}")
    
    return df
