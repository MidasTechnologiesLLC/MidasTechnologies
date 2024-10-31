import argparse
import sys
import time
import scrapers.oil_news_scraper as oil_news
import scrapers.oil_news_preprocessor as oil_news_preprocessor
from tqdm import tqdm

def show_usage_bar(duration):
    for _ in tqdm(range(duration), desc="Processing", unit="sec"):
        time.sleep(1)

def run_scraper():
    print("Starting oil data collection with the scraper...")
    show_usage_bar(0)  # Simulated progress bar duration
    oil_news.run_scraper()
    print("Oil news data scraping completed.")

def run_preprocessor():
    print("Starting oil data collection with the preprocessor...")
    show_usage_bar(0)  # Simulated progress bar duration
    oil_news_preprocessor.run_preprocessor()
    print("Oil news data preprocessing completed.")

def main():
    parser = argparse.ArgumentParser(
        description="Oil News Data Collection Tool"
    )
    parser.add_argument(
        "--scraper", action="store_true", help="Run the oil news scraper (original code)."
    )
    parser.add_argument(
        "--preprocessed", action="store_true", help="Run the oil news preprocessor (new code for sentiment analysis)."
    )

    args = parser.parse_args()

    if args.scraper:
        run_scraper()
    elif args.preprocessed:
        run_preprocessor()
    else:
        print("No valid option selected. Use '--scraper' to run the scraper or '--preprocessed' to run the preprocessor.")
        parser.print_help()

if __name__ == "__main__":
    main()

