import numpy as np             
import yfinance as yf          
import matplotlib.pyplot as plt 
import pandas as pd            

# Downloading historical closing price data for Apple (AAPL) from Yahoo Finance, starting from January 1, 2000
data = yf.download('AAPL', start='2000-01-01')['Close']

# Looping through different window lengths from 1 to 80
for window_len in range(1, 81):

    # Calculating weights for Weighted Moving Average (WMA)
    weights = np.linspace(1, 20, window_len)
    # Normalizing the weights so that they sum to 1
    norm_values_wma = weights / np.sum(weights)

    # Calculating weights for Simple Moving Average (SMA)
    sma_weights = np.ones(window_len) / window_len


# Plotting the WMA and SMA weights
plt.plot(norm_values_wma, label=f'WMA')  
plt.plot(sma_weights, label=f'SMA')      
plt.xlabel('Window Size')                
plt.ylabel('Weights')                    
plt.title('Distribution of SMA and WMA Weights')  
plt.legend()                             
plt.show()                              
