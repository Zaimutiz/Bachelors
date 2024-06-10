import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_days = 20000  # Number of trading days in a year
initial_price = 1000  # Starting price of the stock
mean_return = 0.0005  # Average daily return
volatility = 0.01  # Daily volatility (standard deviation of returns)

# Generate daily returns
daily_returns = np.random.randn(num_days) * volatility + mean_return

# Compute the stock price
stock_prices = initial_price * np.cumprod(1 + daily_returns)

for window_len2 in range(80, 81, 1):

    accuracy_hist = []
    smoothness_hist = []
    accuracy_hist_sheet2 = []
    smoothness_hist_sheet2 = []
    print(window_len2)

    weights = np.linspace(1, 20, window_len2)
    norm_values = weights / np.sum(weights)

    weighted_sum = np.convolve(stock_prices, norm_values, mode='valid')
    data2 = stock_prices[window_len2 - 1:]

    accuracy2 = np.mean(np.abs(weighted_sum - data2))
    smoothness2 = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

    accuracy_hist_sheet2.append(accuracy2)
    smoothness_hist_sheet2.append(smoothness2)

    norm_values_old = norm_values.copy()
    for i in range(1000000):
        norm_values = norm_values_old.copy()

        rand_pos = np.random.randint(norm_values.shape[0])
        norm_values[rand_pos] += np.random.randn() * 0.1

        norm_values = norm_values / np.sum(norm_values)

        weighted_sum = np.convolve(stock_prices, norm_values, mode='valid')
        data2 = stock_prices[window_len2 - 1:]

        accuracy_new = np.mean(np.abs(weighted_sum - data2))
        smoothness_new = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

        if accuracy_new > accuracy2:
            continue

        if smoothness_new > smoothness2:
            continue

        smoothness2 = smoothness_new
        norm_values_old = norm_values.copy()
        accuracy_hist_sheet2.append(accuracy_new)
        smoothness_hist_sheet2.append(smoothness_new)

        accuracy_hist.append(accuracy_new)
        smoothness_hist.append(smoothness_new)

    # Plot norm_values_old and save it as an image
    plt.figure()
    plt.plot(norm_values_old)
    plt.title("CMA svorių pasiskirstymas finansinės laiko eilutės duomenis")
    plt.xlabel('Svorių vieta aibėje')
    plt.ylabel('Svoriai')
    plt.savefig(f'11111Rnadomnormvalues{window_len2}.png')
    plt.close()

    # Plot accuracy_hist vs smoothness_hist and save it as an image
    plt.figure()
    plt.plot(accuracy_hist, smoothness_hist)
    plt.xlabel('accuracy')
    plt.ylabel('smoothness')
    # plt.savefig(f'10000AccuracyAndSmoothness_Window{window_len2}.png')
    plt.close()
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(stock_prices)
plt.title("Simulated Stock Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()