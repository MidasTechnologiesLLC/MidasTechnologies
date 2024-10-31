import json
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import time
import re

OIL_NEWS_URL = "https://oilprice.com/Latest-Energy-News/World-News/"
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_existing_data(file_path):
    """Load existing data from JSON file to avoid duplicates."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_to_json(data, file_path):
    """Save scraped data to a JSON file, ensuring no duplicates."""
    existing_data = load_existing_data(file_path)
    existing_links = {article['link'] for article in existing_data}

    new_data = []
    for article in data:
        if article['link'] in existing_links:
            print(f"Skipping duplicate article: {article['headline']}")
            continue
        new_data.append(article)

    combined_data = existing_data + new_data

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Oil news data saved to {file_path}")

def extract_keywords(text):
    """Improved placeholder function to extract keywords from text."""
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if len(word) > 3]  # Example filter: words longer than 3 chars
    return list(set(keywords))[:10]  # Return up to 10 unique keywords

def analyze_sentiment(text):
    """Placeholder function for sentiment analysis."""
    # Basic placeholder logic (to be replaced with actual sentiment analysis)
    if "profit" in text or "rise" in text:
        return "Positive"
    elif "loss" in text or "decline" in text:
        return "Negative"
    else:
        return "Neutral"

def scrape_oil_news():
    print("Scraping oil market news using Selenium...")

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    news_data = []
    page_number = 1
    max_pages = 10  # Limit to 10 pages

    while page_number <= max_pages:
        driver.get(f"{OIL_NEWS_URL}Page-{page_number}.html")
        
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "categoryArticle"))
            )
        except Exception as e:
            print(f"Error: Content did not load properly on page {page_number}.")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        articles = soup.find_all('div', class_='categoryArticle')
        if not articles:
            print(f"No articles found on page {page_number}. Ending pagination.")
            break

        for article in articles:
            headline = article.find('h2', class_='categoryArticle__title').get_text(strip=True) if article.find('h2', class_='categoryArticle__title') else None
            link = article.find('a', href=True)['href'] if article.find('a', href=True) else None
            date = article.find('p', class_='categoryArticle__meta').get_text(strip=True) if article.find('p', class_='categoryArticle__meta') else None
            excerpt = article.find('p', class_='categoryArticle__excerpt').get_text(strip=True) if article.find('p', class_='categoryArticle__excerpt') else None
            author = date.split('|')[-1].strip() if '|' in date else "Unknown Author"
            timestamp = date.split('|')[0].strip() if '|' in date else date
            
            if headline and link and date:
                news_data.append({
                    'headline': headline,
                    'link': link,
                    'date': timestamp,
                    'author': author,
                    'excerpt': excerpt,
                    'keywords': extract_keywords(headline + " " + excerpt if excerpt else headline),
                    'sentiment_analysis': analyze_sentiment(headline + " " + excerpt if excerpt else headline)
                })

        page_number += 1
        time.sleep(2)

    driver.quit()
    return news_data

def run_scraper():
    file_path = os.path.join(DATA_DIR, 'oil_news.json')
    news_data = scrape_oil_news()
    save_to_json(news_data, file_path)

