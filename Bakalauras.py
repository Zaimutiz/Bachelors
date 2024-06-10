import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import time

start_time = time.time()

data = yf.download('AAPL', start='2000-01-01')['Close']

accuracy_hist = []
smoothness_hist = []

for window_len in range(1, 101):

    weights = np.linspace(1, 20, window_len)
    norm_values = weights / np.sum(weights)
    weighted_sum = np.convolve(data, norm_values, mode='valid')
    data2 = data[window_len - 1:]

    accuracy = np.mean(np.abs(weighted_sum - data2))
    smoothness = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

    accuracy_hist.append(accuracy)
    smoothness_hist.append(smoothness)

results_df = pd.DataFrame({
    'window_len': list(range(1, 101)),
    'accuracy': accuracy_hist,
    'smoothness': smoothness_hist
})

simulations = 1000000
# Modify this path to specify the directory where you want to save the Excel file
excel_file_path = f'AAPLResults_SimulationsNEWEST{simulations}.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(excel_file_path) as writer:

    results_df.to_excel(writer, sheet_name='WMA s and a', index=False)

    for window_len2 in range(1, 101, 1):

        accuracy_hist = []
        smoothness_hist = []
        accuracy_hist_sheet2 = []
        smoothness_hist_sheet2 = []
        print(window_len2)

        weights = np.linspace(1, 20, window_len2)
        norm_values = weights / np.sum(weights)

        weighted_sum = np.convolve(data, norm_values, mode='valid')
        data2 = data[window_len2 - 1:]

        accuracy2 = np.mean(np.abs(weighted_sum - data2))
        smoothness2 = np.mean(np.abs(np.diff(np.diff(weighted_sum))))

        accuracy_hist_sheet2.append(accuracy2)
        smoothness_hist_sheet2.append(smoothness2)

        norm_values_old = norm_values.copy()
        for i in range(simulations):
            norm_values = norm_values_old.copy()

            rand_pos = np.random.randint(norm_values.shape[0])
            norm_values[rand_pos] += np.random.randn() * 0.1

            norm_values = norm_values / np.sum(norm_values)

            weighted_sum = np.convolve(data, norm_values, mode='valid')
            data2 = data[window_len2 - 1:]

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
        plt.title("CMA svorių pasiskirstymas finansinės laiko eilutės duomenis (AAPL) ")
        plt.xlabel('Svorių vieta aibėje')
        plt.ylabel('Svoriai')
        plt.savefig(f'Aplenormvalues{window_len2}.png')
        plt.close()

        # Plot accuracy_hist vs smoothness_hist and save it as an image
        plt.figure()
        plt.plot(accuracy_hist, smoothness_hist)
        plt.xlabel('accuracy')
        plt.ylabel('smoothness')
        #plt.savefig(f'10000AccuracyAndSmoothness_Window{window_len2}.png')
        plt.close()

        results_sheet2_df = pd.DataFrame({
            'window_len': [window_len2] * len(accuracy_hist_sheet2),
            'accuracy_new': accuracy_hist_sheet2,
            'smoothness_new': smoothness_hist_sheet2
        })

        # Write the DataFrame to a sheet named 'WindowSize={window_len2+1}'
        results_sheet2_df.to_excel(writer, sheet_name=f'WindowSize{window_len2}', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
