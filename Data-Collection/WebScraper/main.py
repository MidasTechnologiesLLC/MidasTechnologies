# main.py
import scrapers.oil_news_scraper as oil_news

def main():
    print("Starting oil data collection...")

    # Run oil market news scraper
    oil_news.run_scraper()

    print("Oil news data scraping completed.")

if __name__ == "__main__":
    main()

