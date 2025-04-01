import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Simulation parameters
num_traders = np.linspace(10, 100, 10, dtype=int)  # Trader counts

buyer_types = ['ZIP']
seller_types = buyer_types

# Window length percentages to evaluate
window_percents = [0.05, 0.10, 0.15, 0.20]

# Store results: {num_traders -> {window_percent -> entropy}}
entropy_results = {n: {wp: [] for wp in window_percents} for n in num_traders}

# Iterate over number of traders
for n in num_traders:
    avg_entropy_per_window = {wp: [] for wp in window_percents}

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
                bins = np.linspace(trades_df['returns'].min()*2, trades_df['returns'].max()*2, 500)
                trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

                # Compute rolling entropy for different window sizes
                for wp in window_percents:
                    w = max(10, int(wp * num_trades))  # Ensure minimum window size of 10
                    trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=w).apply(
                        lambda x: calculate_price_entropy(x.dropna()), raw=False
                    )
                    avg_entropy_per_window[wp].append(trades_df['Entropy'].mean())

            except FileNotFoundError:
                print(f"File {trades_file} not found, skipping...")
                continue

    # Store the average entropy per window percentage
    for wp in window_percents:
        entropy_results[n][wp] = np.nanmean(avg_entropy_per_window[wp])

# Plot results
plt.figure(figsize=(10, 6))

for wp in window_percents:
    entropy_values = [entropy_results[n][wp] for n in num_traders]
    plt.plot(num_traders, entropy_values, marker='o', linestyle='-', label=f'{int(wp * 100)}% Window')

plt.xlabel('Number of Traders')
plt.ylabel('Average Entropy')
plt.title('Effect of Rolling Window Length on Entropy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
