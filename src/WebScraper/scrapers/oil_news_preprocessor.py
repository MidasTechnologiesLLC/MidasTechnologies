import json
import re
import os
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from tqdm import tqdm  # Progress bar

OIL_NEWS_URL = "https://oilprice.com/Latest-Energy-News/World-News/"
SCRAPER_DIR = os.path.dirname(os.path.dirname(__file__))  # One level up
DATA_DIR = os.path.join(SCRAPER_DIR, "data")
KEYWORD_FILE_PATH = os.path.join(SCRAPER_DIR, "assets", "oil_key_words.txt")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_to_json(data, file_path):
    existing_data = load_existing_data(file_path)
    existing_links = {article['link'] for article in existing_data if 'link' in article}

    new_data = []
    for article in data:
        if 'link' not in article or article['link'] in existing_links:
            print(f"Skipping duplicate or missing link article: {article.get('headline', 'Unknown Headline')}")
            continue
        new_data.append(article)

    combined_data = existing_data + new_data

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_path}")

def load_keyword_importance(file_path):
    keyword_importance = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    keyword, importance = parts
                    keyword_importance[keyword.lower()] = int(importance)
    else:
        print(f"Keyword file not found at {file_path}")
    return keyword_importance

keyword_importance = load_keyword_importance(KEYWORD_FILE_PATH)

def extract_keywords(text, keyword_importance):
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = {word: keyword_importance[word] for word in words if word in keyword_importance}
    return sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]

def filter_content(content):
    """Remove advertisements, irrelevant phrases, headers, and disclaimers from content."""
    patterns = [
        r'ADVERTISEMENT',                        
        r'Click Here for \d+\+ Global Oil Prices',  
        r'Find us on:',                          
        r'Back to homepage',                     
        r'Join the discussion',                  
        r'More Top Reads From Oilprice.com',     
        r'©OilPrice\.com.*?educational purposes', 
        r'A Media Solutions.*?Oilprice.com',     
        r'\"It\'s most important 8 minute read of my week…\"',  
        r'^[\w\s]*?is a [\w\s]*? for Oilprice\.com.*?More Info',  
        r'^.*?DNOW is a supplier.*?,',             
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    content = re.sub(r'\s+', ' ', content).strip()
    return content

def scrape_author_info(driver, author_url, headline_pages=1):
    """Scrape author's name, bio, contributor since date, and latest article headlines with excerpts, keywords, and timestamp."""
    author_name = "Unknown"
    author_bio = ""
    contributor_since = ""
    other_articles = []

    try:
        # Load author page
        driver.get(author_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        page_source = driver.page_source
        bio_soup = BeautifulSoup(page_source, "html.parser")

        # Extract author name
        author_name_tag = bio_soup.find('h1')
        author_name = author_name_tag.get_text(strip=True) if author_name_tag else "Unknown Author"

        # Extract author bio
        author_bio_tag = bio_soup.find('div', class_='biography')
        author_bio = author_bio_tag.get_text(strip=True) if author_bio_tag else "No bio available"

        # Extract contributor since date
        contributor_since_tag = bio_soup.find('p', class_='contributor_since')
        contributor_since = contributor_since_tag.get_text(strip=True).replace("Contributor since: ", "") if contributor_since_tag else "Unknown Date"

        # Extract latest articles by author with heading, excerpt, keywords, and timestamp
        for page in range(1, headline_pages + 1):
            driver.get(f"{author_url}/Page-{page}.html")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "articles"))
            )
            page_soup = BeautifulSoup(driver.page_source, "html.parser")
            article_tags = page_soup.find_all('li', class_='clear')
            
            for article in article_tags:
                heading_tag = article.find('h3')
                excerpt_tag = article.find('p', class_='articlecontent')
                timestamp_tag = article.find('div', class_='meta')

                if heading_tag and excerpt_tag and timestamp_tag:
                    heading = heading_tag.get_text(strip=True)
                    excerpt = filter_content(excerpt_tag.get_text(strip=True))  # Use filter_content
                    timestamp = timestamp_tag.get_text(strip=True).split("|")[0].replace("Published ", "").strip()
                    keywords = [keyword for keyword, _ in extract_keywords(excerpt, keyword_importance)]
                    
                    other_articles.append({
                        "heading": heading,
                        "excerpt": excerpt,
                        "keywords": keywords,
                        "published_date": timestamp
                    })

    except Exception as e:
        print(f"Error scraping author info: {e}")
        author_name = "Error Occurred"
        author_bio = str(e)
        contributor_since = "N/A"
        other_articles = [{"heading": "Error retrieving articles", "excerpt": "", "keywords": [], "published_date": ""}]

    return {
        "name": author_name,
        "bio": author_bio,
        "contributor_since": contributor_since,
        "other_articles": other_articles
    }

def scrape_oil_news():
    print("Scraping oil news articles for sentiment analysis...")

    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

    news_data = []
    page_number = 1
    max_pages = 1
    total_articles = 0

    while page_number <= max_pages:
        driver.get(f"{OIL_NEWS_URL}Page-{page_number}.html")
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "categoryArticle"))
            )
        except:
            break
        soup = BeautifulSoup(driver.page_source, "html.parser")
        total_articles += len(soup.find_all('div', class_='categoryArticle'))
        page_number += 1

    page_number = 1
    with tqdm(total=total_articles, desc="Scraping articles", unit="article") as pbar:
        while page_number <= max_pages:
            print(f"\nProcessing page {page_number}...")
            driver.get(f"{OIL_NEWS_URL}Page-{page_number}.html")
            soup = BeautifulSoup(driver.page_source, "html.parser")
            articles = soup.find_all('div', class_='categoryArticle')
            if not articles:
                break

            for article in articles:
                headline = article.find('h2', class_='categoryArticle__title').get_text(strip=True) if article.find('h2', class_='categoryArticle__title') else None
                link_tag = article.find('a', href=True)
                link = link_tag['href'] if link_tag else None
                date_meta = article.find('p', class_='categoryArticle__meta')
                date = date_meta.get_text(strip=True).split('|')[0].strip() if date_meta else None
                
                content = ""
                if link:
                    print(f"Fetching article: {link}")
                    driver.get(link)
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "singleArticle"))
                        )
                        article_soup = BeautifulSoup(driver.page_source, "html.parser")
                        raw_content = " ".join([p.get_text(strip=True) for p in article_soup.find_all('p')])
                        content = filter_content(raw_content)
                        
                        # Fetch author info using scrape_author_info
                        author_url = article_soup.find('a', text=re.compile(r'More Info|Read More', re.IGNORECASE))['href']
                        author_info = scrape_author_info(driver, author_url, headline_pages=1)
                        
                    except:
                        print(f"Error: Content did not load for article {headline}.")
                        author_info = {
                            "name": "Unknown",
                            "bio": "",
                            "contributor_since": "",
                            "other_articles": []
                        }
                
                extracted_keywords = extract_keywords(f"{headline} {content}", keyword_importance)

                if headline and link and date:
                    news_data.append({
                        'headline': headline,
                        'link': link,
                        'content': content,
                        'date': date,
                        'author': author_info['name'],
                        'author_bio': author_info['bio'],
                        'contributor_since': author_info['contributor_since'],
                        'other_articles': author_info['other_articles'],
                        'keywords': extracted_keywords,
                    })

                pbar.set_postfix_str(f"Processing article: {headline[:40]}...")
                pbar.update(1)

            page_number += 1
            time.sleep(2)

    driver.quit()
    return news_data

def run_preprocessor():
    file_path = os.path.join(DATA_DIR, 'preprocessed_oil_news.json')
    news_data = scrape_oil_news()
    save_to_json(news_data, file_path)

if __name__ == "__main__":
    run_preprocessor()

