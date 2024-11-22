from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def scrape_opentable_reviews(url, max_reviews=10):
    """
    Scrapes reviews from an OpenTable restaurant page.

    Args:
    - url (str): The URL of the restaurant's OpenTable page.
    - max_reviews (int): The maximum number of reviews to scrape.

    Returns:
    - list of dict: A list of reviews with the review text and date.
    """
    # Initialize WebDriver
    driver = webdriver.Chrome()  # Ensure `chromedriver` is installed and in your PATH
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    reviews = []

    try:
        # Scroll and load more reviews until reaching the desired count or no more reviews
        while len(reviews) < max_reviews:
            # Extract review elements
            review_elements = driver.find_elements(By.CLASS_NAME, 'afkKaa-4T28-')

            for review_element in review_elements:
                try:
                    review_text = review_element.find_element(By.CLASS_NAME, '_6rFG6U7PA6M-').text
                    review_date = review_element.find_element(By.CLASS_NAME, 'iLkEeQbexGs-').text
                    reviews.append({'review_text': review_text, 'review_date': review_date})

                    # Stop if we reach the maximum reviews
                    if len(reviews) >= max_reviews:
                        break
                except Exception as e:
                    print(f"Error extracting review: {e}")
                    continue

            # Attempt to click "Load More Reviews" if it exists
            try:
                load_more_button = driver.find_element(By.CSS_SELECTOR, '.ot-load-more-reviews')
                load_more_button.click()
                time.sleep(3)  # Wait for reviews to load
            except:
                print("No more reviews to load.")
                break

    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        driver.quit()

    return reviews


# Example usage
if __name__ == "__main__":
    opentable_url = "https://www.opentable.com/r/sacre-frenchy-paris?corrid=400c4ade-c1da-4fb9-88a7-a165c57bab08&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&p=2&sd=2024-11-22T19%3A00%3A00"  # Replace with the actual URL
    scraped_reviews = scrape_opentable_reviews(opentable_url, max_reviews=20)

    # Print the reviews
    for idx, review in enumerate(scraped_reviews, 1):
        print(f"Review {idx}:")
        print(f"Date: {review['review_date']}")
        print(f"Text: {review['review_text']}\n")