import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

stock_name='AMZN'
df = pd.read_csv(f'data/{stock_name}_tweets.csv')
df.head()


sent_df = df.copy()
sent_df["sentiment_score"] = ''
sent_df["Negative"] = ''
sent_df["Neutral"] = ''
sent_df["Positive"] = ''
sent_df.head()

sentiment_i_a = SentimentIntensityAnalyzer()
for indx, row in sent_df.iterrows():  # Use iterrows() for iterating over rows
    try:
        sentence_i = unicodedata.normalize('NFKD', row['Tweet'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        sent_df.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
        sent_df.at[indx, 'Negative'] = sentence_sentiment['neg']
        sent_df.at[indx, 'Neutral'] = sentence_sentiment['neu']
        sent_df.at[indx, 'Positive'] = sentence_sentiment['pos']
    except TypeError:
        print(f"Error processing row {indx}: {row['Tweet']}")
        break

sent_df['Date'] = pd.to_datetime(sent_df['Date'])
sent_df['Date'] = sent_df['Date'].dt.date
sent_df = sent_df.drop(columns=['Negative', 'Positive', 'Neutral'])
# Remove the 'Tweet' column
sent_df = sent_df.drop(columns=['Tweet'])

# Group by 'Date' and calculate the mean of 'sentiment_score'
twitter_df = sent_df.groupby('Date', as_index=False).mean()

# Save the resulting DataFrame
twitter_df.to_csv(f'data/{stock_name}_twitter_sentiment.csv', index=False)