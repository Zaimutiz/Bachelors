import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define the ticker symbol for Bitcoin and the time period
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-12-31'

# Download the Bitcoin data
bitcoin_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate the 20-day SMA
bitcoin_data['SMA_20'] = bitcoin_data['Close'].rolling(window=20).mean()

# Plot the closing prices and SMA
plt.figure(figsize=(10, 5))
plt.plot(bitcoin_data['Close'], label='APPL uždarymo kaina')
plt.plot(bitcoin_data['SMA_20'], label='SMA', color='orange')
plt.title('SMA pagal "Apple Inc." finansinės laiko eilutės duomenis nuo 2020 metų')
plt.xlabel('Data')
plt.ylabel('Uždarymo Kaina')
plt.legend()
plt.show()
