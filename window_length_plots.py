import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


# Simulation parameters
num_traders = np.linspace(10, 100, 10, dtype=int)  # Trader counts
buyer_types = ['ZIP', 'ZIC', 'GVWY', 'SHVR']  # Multiple buyer types
seller_types = ['GVWY']

# Extended window length percentages to evaluate
window_percents = np.linspace(10, 99, 10, dtype=int)/100

# Store results: {window_percent -> {buyer_type -> [entropy values]}}
entropy_results = {wp: {bt: [] for bt in buyer_types} for wp in window_percents}

# Iterate over number of traders
for n in num_traders:
    for wp in window_percents:
        avg_entropy_per_buyer = {bt: [] for bt in buyer_types}

        # Iterate over trader types
        for seller in seller_types:
            for buyer in buyer_types:
                trial_id = f'{n}_{buyer}B_{seller}S'
                trades_file = f'{trial_id}_tape.csv'

                try:
                    # Read trade data
                    trades_df = pd.read_csv(trades_file)
                    trades_df.columns = ['Type', 'Time', 'Price']
                    trades_df = trades_df.sort_values(by="Time")
                    trades_df['returns'] = trades_df['Price'].pct_change().dropna()

                    # Total number of trades
                    num_trades = len(trades_df)

                    # Bin returns
                    bins = np.linspace(trades_df['returns'].min() * 2, trades_df['returns'].max() * 2, 500)
                    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

                    # Compute rolling entropy
                    w = max(10, int(wp * num_trades))  # Ensure minimum window size of 10
                    trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=w).apply(
                        lambda x: calculate_price_entropy(x.dropna()), raw=False
                    )
                    avg_entropy_per_buyer[buyer].append(trades_df['Entropy'].mean())

                except FileNotFoundError:
                    print(f"File {trades_file} not found, skipping...")
                    continue

        # Store the average entropy per buyer type
        for bt in buyer_types:
            entropy_results[wp][bt].append(np.nanmean(avg_entropy_per_buyer[bt]))

# Determine subplot layout
n_rows = 3
n_cols = int(np.ceil(len(window_percents) / n_rows))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
axes = axes.flatten()

for ax, wp in zip(axes, window_percents):
    for bt in buyer_types:
        ax.plot(num_traders, entropy_results[wp][bt], marker='o', linestyle='-', label=f'{bt}')

    ax.set_ylabel(f'Entropy (Window {int(wp * 100)}%)')
    ax.legend()
    ax.grid()
    ax.set_title(f'Entropy vs. Traders (Window {int(wp * 100)}%)')

axes[-1].set_xlabel('Number of Traders')
plt.tight_layout()
plt.show()
