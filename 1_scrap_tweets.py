import asyncio
from twikit import Client
import pandas as pd

# Initialize the Twikit client with language preference
client = Client('en-US')

# Twitter account credentials (avoid hardcoding sensitive credentials in production)
USERNAME = 'omkardc18'
EMAIL = 'omkardc18@gmail.com'
PASSWORD = 'oDc@1919'

# Function to log in to Twitter
async def login():
    """
    Logs in to Twitter using Twikit client.
    """
    print("Logging in...")
    await client.login(auth_info_1=USERNAME, auth_info_2=EMAIL, password=PASSWORD)
    print("Login successful!")

# Function to scrape tweets based on a keyword
async def scrape_tweets(keyword, count=100):
    """
    Scrapes tweets for a given keyword.

    Args:
    - keyword (str): The keyword to search for.
    - count (int): Number of tweets to retrieve.

    Returns:
    - list: A list of dictionaries with tweet data.
    """
    print(f"Scraping {count} tweets for keyword: {keyword}")
    tweets = await client.search_tweet(keyword, 'Latest', count=count)

    # Extract relevant tweet data
    tweet_data = []
    for tweet in tweets:
        tweet_data.append({
            'Date': tweet.created_at,
            'ID': tweet.id,
            'Content': tweet.text,
            'Username': tweet.user.name
        })

    print(f"Scraped {len(tweet_data)} tweets.")
    return tweet_data

# Function to save tweets to a CSV file
def save_to_csv(tweet_data, filename):
    """
    Saves tweet data to a CSV file.

    Args:
    - tweet_data (list): List of dictionaries containing tweet data.
    - filename (str): Name of the output CSV file.
    """
    df = pd.DataFrame(tweet_data)
    df.to_csv(filename, index=False)
    print(f"Saved data to {filename}")

# Main function to execute the scraping and saving process
if __name__ == "__main__":
    keyword = "AMZN"  # Replace with the desired keyword or stock symbol
    count = 100       # Number of tweets to scrape
    output_file = f"{keyword}_tweets.csv"

    async def main():
        await login()
        tweet_data = await scrape_tweets(keyword, count)
        if tweet_data:
            save_to_csv(tweet_data, output_file)
        else:
            print("No tweets found!")

    # Run the main asynchronous function
    asyncio.run(main())
