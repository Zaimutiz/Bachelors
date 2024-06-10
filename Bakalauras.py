import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import time

# Record the start time to measure the elapsed time
start_time = time.time()

# Download historical closing price data for Apple (AAPL) starting from January 1, 2000
data = yf.download('AAPL', start='2000-01-01')['Close']

# Lists to store accuracy and smoothness values
accuracy_hist = []
smoothness_hist = []

# Loop over different window lengths from 1 to 100
for window_len in range(1, 101):

    # Calculate weights for the weighted moving average (WMA)
    weights = np.linspace(1, 20, window_len)
    norm_values = weights / np.sum(weights)  # Normalize the weights so they sum to 1
    weighted_sum = np.convolve(data, norm_values, mode='valid')  # Apply the weights using convolution
    data2 = data[window_len - 1:]  # Align the data to the length of the weighted sum

    # Calculate accuracy (mean absolute error) and smoothness (mean absolute second difference)
    accuracy = np.mean(np.abs(weighted_sum - data2))
    smoothness = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

    # Store accuracy and smoothness values in their respective lists
    accuracy_hist.append(accuracy)
    smoothness_hist.append(smoothness)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'window_len': list(range(1, 101)),
    'accuracy': accuracy_hist,
    'smoothness': smoothness_hist
})

# Number of simulations to run for weight optimization
simulations = 1000000
# Specify the path to save the Excel file
excel_file_path = f'AAPLResults_SimulationsNEWEST{simulations}.xlsx'

# Create an Excel writer using Pandas and XlsxWriter
with pd.ExcelWriter(excel_file_path) as writer:

    # Write the initial results DataFrame to the Excel file
    results_df.to_excel(writer, sheet_name='WMA s and a', index=False)

    # Loop over different window lengths again to optimize weights
    for window_len2 in range(1, 101, 1):

        # Lists to store accuracy and smoothness values for the current window length
        accuracy_hist = []
        smoothness_hist = []
        accuracy_hist_sheet2 = []
        smoothness_hist_sheet2 = []
        print(window_len2)

        # Calculate initial weights for the weighted moving average (WMA)
        weights = np.linspace(1, 20, window_len2)
        norm_values = weights / np.sum(weights)

        # Calculate the initial weighted sum
        weighted_sum = np.convolve(data, norm_values, mode='valid')
        data2 = data[window_len2 - 1:]

        # Calculate initial accuracy and smoothness
        accuracy2 = np.mean(np.abs(weighted_sum - data2))
        smoothness2 = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

        # Store initial accuracy and smoothness values in their respective lists
        accuracy_hist_sheet2.append(accuracy2)
        smoothness_hist_sheet2.append(smoothness2)

        # Copy of the normalized weights for optimization
        norm_values_old = norm_values.copy()
        
        # Optimization loop to adjust weights
        for i in range(simulations):
            norm_values = norm_values_old.copy()

            # Randomly select a position in the weight array and perturb it
            rand_pos = np.random.randint(norm_values.shape[0])
            norm_values[rand_pos] += np.random.randn() * 0.1

            # Normalize the perturbed weights so they sum to 1
            norm_values = norm_values / np.sum(norm_values)

            # Recalculate the weighted sum with the new weights
            weighted_sum = np.convolve(data, norm_values, mode='valid')
            data2 = data[window_len2 - 1:]

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
        plt.title("CMA Weight Distribution for Financial Time Series Data (AAPL)")
        plt.xlabel('Weight Position in Array')
        plt.ylabel('Weights')
        plt.savefig(f'Aplenormvalues{window_len2}.png')
        plt.close()

        # Plot accuracy versus smoothness and save the plot as an image
        plt.figure()
        plt.plot(accuracy_hist, smoothness_hist)
        plt.xlabel('Accuracy')
        plt.ylabel('Smoothness')
        #plt.savefig(f'10000AccuracyAndSmoothness_Window{window_len2}.png')
        plt.close()

        # Create a DataFrame to store the results for the current window length
        results_sheet2_df = pd.DataFrame({
            'window_len': [window_len2] * len(accuracy_hist_sheet2),
            'accuracy_new': accuracy_hist_sheet2,
            'smoothness_new': smoothness_hist_sheet2
        })

        # Write the DataFrame to a new sheet in the Excel file
        results_sheet2_df.to_excel(writer, sheet_name=f'WindowSize{window_len2}', index=False)

# Record the end time and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
