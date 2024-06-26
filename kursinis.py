# -*- coding: utf-8 -*-
"""Kursinis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R6X1KatXEzbgN6_nzHGNvddhyq3-Tx9L
"""

# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Function to get historical stock data from Yahoo Finance
def get_yahoo_finance_data(symbol, start_date, end_date):
    try:
        # Download historical data from Yahoo Finance for the specified symbol and date range
        data = yf.download(symbol, start=start_date, end=end_date)
        # Select relevant columns (Volume, Close, High, Low, Adj Close) and forward-fill missing values
        return data[['Volume', 'Close', 'High', 'Low', 'Adj Close']].ffill()
    except Exception as e:
        # Handle exceptions if there is an error fetching data
        print(f"Error fetching data for {symbol} from Yahoo Finance: {e}")
        return None

# Function to calculate moving averages and related metrics
def calculate_moving_average(df, window, method):
    if method == 'SMA':
        # Calculate Simple Moving Average (SMA) using the specified window size
        rolling_avg = df['Close'].rolling(window=window).mean()
    elif method == 'EMA':
        # Calculate Exponential Moving Average (EMA) using the specified window size
        rolling_avg = df['Close'].ewm(span=window, adjust=False).mean()
    elif method == 'WMA':
        # Calculate Weighted Moving Average (WMA) using the specified window size and weights
        weights = pd.Series(range(1, window + 1))
        rolling_avg = df['Close'].rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    elif method == 'TMA':
        # Calculate Double Moving Average (TMA) by applying SMA twice
        rolling_avg = df['Close'].rolling(window=window).mean().rolling(window=window).mean()
    elif method == 'VWMA':
        # Calculate Volume Weighted Moving Average (VWMA)
        weighted_volume_avg = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
        rolling_avg = weighted_volume_avg
    else:
        # Raise an error for unsupported methods
        raise ValueError(f"Unsupported method: {method}")

    # Calculate inaccuracy, roughness, and final roughness
    df[f'{method}_Inaccuracy'] = abs(df['Close'] - rolling_avg)
    df[f'{method}_Roughness'] = abs(rolling_avg.diff())
    df[f'Final_{method}_Roughness'] = abs(df[f'{method}_Roughness'].diff())

    # Return a dictionary containing window size and average metrics
    return {
        'Window': window,
        f'Average_{method}_Inaccuracy': df[f'{method}_Inaccuracy'].mean(),
        f'Average_{method}_Roughness': df[f'Final_{method}_Roughness'].mean()
    }

# Function to calculate moving averages for different methods and plot the results
def calculate_and_plot_averages(df):
    # List of moving average methods to be considered
    methods = ['SMA', 'EMA', 'WMA', 'TMA', 'VWMA']
    # Dictionary to store results for each method
    results = {method: [] for method in methods}

    # Iterate through different window sizes
    for window in range(1, 201):
        for method in methods:
            # Calculate moving averages and related metrics for each method and window size
            averages = calculate_moving_average(df.copy(), window, method)
            results[method].append(averages)

    # Create DataFrames from the results for each method
    averages_df = {method: pd.DataFrame(results[method]) for method in methods}

    # Plot the data for each method
    for method in methods:
        plt.plot(averages_df[method][f'Average_{method}_Inaccuracy'], averages_df[method][f'Average_{method}_Roughness'], label=f'{method}')

    # Set plot labels and title
    plt.xlabel('Šiurkštumas')
    plt.ylabel('Neatitikimas')
    plt.title('Neatitikimo ir šiurkštumo reikšmės remiantis augančiu apimties dydžiu ')
    plt.legend()
    plt.show()

    # Return DataFrames containing average metrics for each method
    return averages_df

# Function to save average values to CSV
def save_averages_to_csv(averages_df, method):
    # Define the output file path based on the method
    output_file_path = f'D:/output_data_averages_{method.lower()}.csv'
    # Save the DataFrame to a CSV file without index
    averages_df.to_csv(output_file_path, index=False)

# Specify the stock symbol and date range
stock_symbol = 'BTC-USD'
start_date = '2022-01-01'
end_date = '2023-12-01'

# Get historical stock data from Yahoo Finance
yahoo_data = get_yahoo_finance_data(stock_symbol, start_date, end_date)

if yahoo_data is not None:
    # Merge Yahoo Finance data with existing data
    df = pd.DataFrame()
    if not yahoo_data.empty:
        # Merging relevant columns (Volume, Close, High, Low) to the DataFrame
        df = pd.concat([df, yahoo_data[['Volume', 'Close', 'High', 'Low']]], axis=1, join='outer')
        # Drop rows with missing values
        df = df.dropna()
        # Calculate and plot moving average metrics
        averages_df = calculate_and_plot_averages(df)

        # Saving averages to CSV
        # for method in averages_df:
        #    save_averages_to_csv(averages_df[method], method)
else:
    print(f"Error getting data from Yahoo Finance.")

