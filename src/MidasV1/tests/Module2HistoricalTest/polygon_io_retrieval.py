import requests
import csv
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Polygon API configuration
API_KEY = "RumpAnkjWlY69jKygFzMyDm5Chc3luDr"  # Replace with your Polygon.io API key
BASE_URL = "https://api.polygon.io/v3/reference/tickers"
OUTPUT_FILE = "data/us_tickers.csv"

# API parameters
params = {
    "market": "stocks",  # US stock market
    "active": "true",    # Only active tickers
    "limit": 1000,       # Max limit per request
    "apiKey": API_KEY
}

def fetch_tickers():
    tickers = []
    next_url = BASE_URL

    while next_url:
        logging.info(f"Fetching data from {next_url}")
        response = requests.get(next_url, params=params)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            tickers.extend(results)

            # Check if there are more pages
            next_url = data.get("next_url", None)
            time.sleep(1)  # Rate limit handling (1 second delay)
        else:
            logging.error(f"Failed to fetch data: {response.status_code} - {response.text}")
            break

    return tickers

def save_to_csv(tickers):
    # Save the tickers to a CSV file
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Ticker", "Name", "Market", "Locale", "Primary Exchange", "Type", "Currency"])
        # Write rows
        for ticker in tickers:
            writer.writerow([
                ticker.get("ticker"),
                ticker.get("name"),
                ticker.get("market"),
                ticker.get("locale"),
                ticker.get("primary_exchange"),
                ticker.get("type"),
                ticker.get("currency")
            ])
    logging.info(f"Saved {len(tickers)} tickers to {OUTPUT_FILE}")

def main():
    logging.info("Starting ticker retrieval program...")

    # Fetch tickers
    tickers = fetch_tickers()
    if tickers:
        logging.info(f"Fetched {len(tickers)} tickers.")
        save_to_csv(tickers)
    else:
        logging.warning("No tickers fetched. Exiting program.")

if __name__ == "__main__":
    main()

