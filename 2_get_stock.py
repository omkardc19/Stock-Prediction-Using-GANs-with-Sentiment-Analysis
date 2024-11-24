import yfinance as yf
import pandas as pd
import os

# Define the stock name
stock_name = "AMZN"

# Download stock data from Yahoo Finance for the specified date range
stock_df = yf.download(stock_name, start="2021-09-30", end="2022-09-30")

# Drop the multi-level column index if it exists
stock_df.columns = stock_df.columns.droplevel(1)

# Select relevant columns
stock_df = stock_df[['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]

# Reset the index of the DataFrame
stock_df.reset_index(inplace=True)

# Remove the 'Price' column if it exists
if 'Price' in stock_df.columns:
    stock_df.drop(columns=['Price'], inplace=True)

# Define the output path for the CSV file
output_path = os.path.join("C:/Users/Omkar/Desktop/Stock", f"{stock_name}_data.csv")

# Save the DataFrame to a CSV file without the index
stock_df.to_csv(output_path, index=False)