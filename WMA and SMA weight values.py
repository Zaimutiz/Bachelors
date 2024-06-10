import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

data = yf.download('AAPL', start='2000-01-01')['Close']

for window_len in range(1, 81):

    # Calculating WMA weights
    weights = np.linspace(1, 20, window_len)
    norm_values_wma = weights / np.sum(weights)

    # Calculating SMA weights
    sma_weights = np.ones(window_len) / window_len


# Plotting both WMA and SMA weights
plt.plot(norm_values_wma, label=f'WMA')
plt.plot(sma_weights, label=f'SMA')
plt.xlabel('Apimties dydis')
plt.ylabel('Svoriai')
plt.title('SMA ir WMA svorių pasiskirstymas')
plt.legend()
plt.show()