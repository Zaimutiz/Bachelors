import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_days = 20000  # Number of trading days in the simulation
initial_price = 1000  # Starting price of the stock
mean_return = 0.0005  # Average daily return
volatility = 0.01  # Daily volatility (standard deviation of returns)

# Generate daily returns
daily_returns = np.random.randn(num_days) * volatility + mean_return

# Compute the stock price by cumulatively multiplying the daily returns
stock_prices = initial_price * np.cumprod(1 + daily_returns)

# Loop over window lengths, here it's effectively a single iteration
for window_len2 in range(80, 81, 1):

    # Lists to store accuracy and smoothness history for plotting
    accuracy_hist = []
    smoothness_hist = []
    accuracy_hist_sheet2 = []
    smoothness_hist_sheet2 = []
    print(window_len2)

    # Calculate initial weights for the weighted moving average (WMA)
    weights = np.linspace(1, 20, window_len2)
    norm_values = weights / np.sum(weights)

    # Calculate the weighted sum using convolution, which applies the weights to the stock prices
    weighted_sum = np.convolve(stock_prices, norm_values, mode='valid')
    data2 = stock_prices[window_len2 - 1:]

    # Calculate initial accuracy (mean absolute error) between weighted sum and actual stock prices
    accuracy2 = np.mean(np.abs(weighted_sum - data2))
    # Calculate initial smoothness (mean absolute second difference) of the weighted sum
    smoothness2 = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

    # Store initial accuracy and smoothness in the history lists
    accuracy_hist_sheet2.append(accuracy2)
    smoothness_hist_sheet2.append(smoothness2)

    # Copy of the original normalized weights to use in optimization
    norm_values_old = norm_values.copy()
    # Optimization loop to adjust weights
    for i in range(1000000):
        norm_values = norm_values_old.copy()

        # Randomly select a position in the weight array and perturb it
        rand_pos = np.random.randint(norm_values.shape[0])
        norm_values[rand_pos] += np.random.randn() * 0.1

        # Normalize the perturbed weights so they sum to 1
        norm_values = norm_values / np.sum(norm_values)

        # Recalculate the weighted sum with the new weights
        weighted_sum = np.convolve(stock_prices, norm_values, mode='valid')
        data2 = stock_prices[window_len2 - 1:]

        # Calculate new accuracy and smoothness with the adjusted weights
        accuracy_new = np.mean(np.abs(weighted_sum - data2))
        smoothness_new = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

        # If the new weights do not improve both accuracy and smoothness, skip this iteration
        if accuracy_new > accuracy2:
            continue

        if smoothness_new > smoothness2:
            continue

        # Update smoothness, weights, and history if new values are better
        smoothness2 = smoothness_new
        norm_values_old = norm_values.copy()
        accuracy_hist_sheet2.append(accuracy_new)
        smoothness_hist_sheet2.append(smoothness_new)

        # Add new accuracy and smoothness to the history lists
        accuracy_hist.append(accuracy_new)
        smoothness_hist.append(smoothness_new)

    # Plot the optimized weights and save the plot as an image
    plt.figure()
    plt.plot(norm_values_old)
    plt.title("CMA Weight Distribution for Financial Time Series Data")
    plt.xlabel('Weight Position in Array')
    plt.ylabel('Weights')
    plt.savefig(f'11111Rnadomnormvalues{window_len2}.png')
    plt.close()

    # Plot accuracy versus smoothness and save the plot as an image
    plt.figure()
    plt.plot(accuracy_hist, smoothness_hist)
    plt.xlabel('Accuracy')
    plt.ylabel('Smoothness')
    # plt.savefig(f'10000AccuracyAndSmoothness_Window{window_len2}.png')
    plt.close()

# Plot the simulated stock prices over time
plt.figure(figsize=(10, 5))
plt.plot(stock_prices)
plt.title("Simulated Stock Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
