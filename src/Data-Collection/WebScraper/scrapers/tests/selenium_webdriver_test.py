from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
import time

# Provide the path to your geckodriver executable using the Service class
service = Service(executable_path='/usr/local/bin/geckodriver')
driver = webdriver.Firefox(service=service)

# Open a website (e.g., OilPrice.com)
driver.get("https://oilprice.com/Latest-Energy-News/World-News/")

# Wait for the page to load
time.sleep(5)

# Print the title of the page to verify that it's loaded
print(driver.title)

# Find and print some element on the page, e.g., all article titles
articles = driver.find_elements(By.CSS_SELECTOR, "div.categoryArticle")
for article in articles:
    title = article.find_element(By.TAG_NAME, "a").text
    print(f"Article title: {title}")

# Close the browser
driver.quit()
