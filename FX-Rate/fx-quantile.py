# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def read_csv_to_mat(file_name):
    return pd.read_csv(file_name, index_col=False).groupby(['Trial', 'Parameter']).mean().to_numpy()


def plot_percentile(ax, var):
    percentiles=[0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    var_percentiles = np.percentile(var, percentiles, axis=0)

    median_idx = len(var_percentiles) // 2

    x = np.arange(0, var.shape[1])
    ax.semilogy(x, var_percentiles[median_idx, :], 'k-', label='median')

    for incr in range(1, median_idx+1):
        ax.fill_between(x, var_percentiles[median_idx - incr, :], var_percentiles[median_idx + incr, :], color='b', alpha=0.2 * 0.9 ** incr)
    
    ax.grid(axis='y', linestyle='--')
    # ax.legend()

# %%
# read csv
fx = 1 / read_csv_to_mat('FX_Rate.csv')
fx_orig = 1/ read_csv_to_mat('FX_Rate_Orig.csv')

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
plot_percentile(axs[0], fx_orig)
axs[0].set_title("Corr(USD NIR, HKD NIR) = Moody's assumption", fontsize=10)
axs[0].set_ylabel('USD/HKD FX rate', fontsize=10)
plot_percentile(axs[1], fx)
axs[1].set_title('Corr(USD NIR, HKD NIR) = 0.95', fontsize=10)
axs[1].tick_params(axis='y', which='both', labelleft=False)
fig.suptitle('Percentile plot - USD/HKD FX rate')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.1)
plt.show()


# %%
