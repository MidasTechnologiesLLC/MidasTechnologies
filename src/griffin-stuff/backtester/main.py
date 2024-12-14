import json
import logging
import pandas as pd
import os

from indicators import calculate_indicator_signals
from evaluation import evaluate_indicator_accuracy

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data(csv_path, date_col, price_col):
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[date_col, price_col])
    return df

if __name__ == "__main__":
    config = load_config("config.json")
    setup_logging(config["evaluation"]["log_file"])
    
    # Load data
    df = load_data(config["data"]["input_csv"], 
                   config["data"]["date_column"], 
                   config["data"]["price_column"])
    
    # Calculate indicators and signals, evaluate accuracy
    all_results = []
    for category, indicators in config["indicators"].items():
        for ind_name in indicators:
            params = config["parameters"].get(ind_name, {})
            
            signals_df = calculate_indicator_signals(
                df.copy(),
                indicator_name=ind_name,
                params=params,
                price_col=config["data"]["price_column"],
                high_col=config["data"]["high_column"],
                low_col=config["data"]["low_column"],
                volume_col=config["data"]["volume_column"]
            )
            
            accuracy = evaluate_indicator_accuracy(
                signals_df, 
                price_col=config["data"]["price_column"],
                horizon=config["evaluation"]["prediction_horizon"]
            )
            
            logging.info(f"Category: {category}, Indicator: {ind_name}, Accuracy: {accuracy:.4f}")
            all_results.append((category, ind_name, accuracy))
    
    # Print results to console as well
    for category, ind_name, acc in all_results:
        print(f"Category: {category}, Indicator: {ind_name}, Accuracy: {acc:.4f}")
