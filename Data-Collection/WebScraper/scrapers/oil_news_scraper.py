# scrapers/oil_news_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# URL for OilPrice.com homepage
OIL_NEWS_URL = "https://oilprice.com/Latest-Energy-News/World-News/"

# Define the directory to store the scraped data
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function to scrape news headlines from OilPrice.com
def scrape_oil_news():
    print("Scraping oil market news...")

    # Send an HTTP request to the website
    response = requests.get(OIL_NEWS_URL)
    response.raise_for_status()

    # Print the HTML to see what we are working with
    print(response.text[:1000])  # Print only the first 1000 characters for brevity

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all news article containers (class names updated)
    articles = soup.find_all('div', class_='categoryArticle')

    # List to store the scraped data
    news_data = []

    # Loop through each article container
    for article in articles:
        # Extract the headline, date, and link
        headline = article.find('a').get_text(strip=True) if article.find('a') else None
        link = article.find('a')['href'] if article.find('a') else None
        date = article.find('span', class_='categoryArticle__date').get_text(strip=True) if article.find('span', class_='categoryArticle__date') else None

        # Only append valid data
        if headline and link and date:
            news_data.append({
                'headline': headline,
                'link': f"https://oilprice.com{link}",
                'date': date
            })

    df = pd.DataFrame(news_data)
    return df

# Function to run the scraper and save data
def run_scraper():
    # Scrape oil news
    news_df = scrape_oil_news()

    # Define the file path for saving the data
    file_path = os.path.join(DATA_DIR, 'oil_news.csv')

    # Save the DataFrame to a CSV file
    if not news_df.empty:
        news_df.to_csv(file_path, index=False)
        print(f"Oil news data saved to {file_path}")
    else:
        print("No data was scraped. The CSV file is empty.")
