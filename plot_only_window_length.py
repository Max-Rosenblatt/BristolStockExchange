import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simpleshock_v1 import range1, range2

from BSE import market_session

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Loop over each buyer type
trial_id = '40_ZIP_buyers_vs_ZIP_sellers'

# Fixed parameters
n_bins = 2000
window_length = 50  # Fixed rolling window

window_lengths = np.linspace(5,100,20,dtype=int)

for window_length in window_lengths:
    # Read in market data
    trades_file = f'{trial_id}_tape.csv'
    trades_df = pd.read_csv(trades_file)
    trades_df.columns = ['Type', 'Time', 'Price']
    trades_df['id'] = range(len(trades_df))
    trades_df.set_index('id', inplace=True)
    trades_df = trades_df.sort_values(by="Time")
    trades_df['returns'] = trades_df['Price'].pct_change().dropna()

    # Bin returns and calculate entropy
    bins = np.linspace(trades_df['returns'].min(), trades_df['returns'].max(), n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
    trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=window_length).apply(
        lambda x: calculate_price_entropy(x.dropna()), raw=False)

    # Append to list

    # Create subplots for Price and Entropy
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Scatter plot for trade prices
    axes[0].scatter(trades_df['Time'], trades_df['Price'], marker='x', color='black')
    axes[0].set_ylabel("Trade Price")
    axes[0].set_title(f"Trade Prices Over Time, Window Length {window_length}")
    axes[0].grid()
    axes[0].axvline(x=300, color='black', linestyle='--', linewidth=1.5)

    # Scatter plot for entropy
    axes[1].scatter(trades_df['Time'], trades_df['Entropy'], marker='x', label=f'Entropy', color='blue')
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title("Price Entropy Over Time")
    axes[1].grid()

    axes[0].set_xlim(0, trades_df['Time'].max())
    axes[1].set_xlim(0, trades_df['Time'].max())

    plt.tight_layout()
    plt.show()