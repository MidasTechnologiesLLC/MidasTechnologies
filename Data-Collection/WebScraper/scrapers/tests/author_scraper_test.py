import json
import re
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

AUTHOR_URL = "https://oilprice.com/contributors/Charles-Kennedy"  # Replace with actual author URL
OUTPUT_FILE = "author_info.json"

def extract_keywords(text):
    """Basic keyword extraction by finding unique words longer than 3 characters."""
    words = re.findall(r'\b\w{4,}\b', text.lower())
    keywords = list(set(words))
    return keywords[:10]  # Limit to top 10 unique keywords for simplicity

def scrape_author_info(author_url, headline_pages=1):
    """Scrape author's name, bio, contributor since date, and latest article headlines with excerpts, keywords, and timestamp."""
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options)

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
                    excerpt = excerpt_tag.get_text(strip=True)
                    timestamp = timestamp_tag.get_text(strip=True).split("|")[0].replace("Published ", "").strip()
                    keywords = extract_keywords(excerpt)
                    
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

    finally:
        driver.quit()

    return {
        "name": author_name,
        "bio": author_bio,
        "contributor_since": contributor_since,
        "other_articles": other_articles
    }

def save_to_json(data, output_file):
    """Save author info to a JSON file."""
    with open(output_file, mode="w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Author info saved to {output_file}")

if __name__ == "__main__":
    # Scrape author info
    author_info = scrape_author_info(AUTHOR_URL, headline_pages=1)

    # Save to JSON
    save_to_json(author_info, OUTPUT_FILE)

