import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Simulation parameters
start_time = 0
end_time = 600
num_traders = [20,50,80,100]  # Traders count from first script
window_lengths = np.linspace(10, 1000, 20, dtype=int)  # Different rolling window sizes

buyer_types = ['ZIP']
seller_types = buyer_types

# Store results: { num_traders -> [entropy for each window length] }
entropy_results = {n: [] for n in num_traders}

# Iterate over number of traders
for n in num_traders:
    avg_entropy_per_window = {w: [] for w in window_lengths}

    # Iterate over trader types
    for seller in seller_types:
        for buyer in buyer_types:
            trial_id = f'{n}_{buyer}B_{seller}S'
            trades_file = f'data/{trial_id}_tape.csv'

            try:
                # Read trade data
                trades_df = pd.read_csv(trades_file)
                trades_df.columns = ['Type', 'Time', 'Price']
                trades_df = trades_df.sort_values(by="Time")
                trades_df['returns'] = trades_df['Price'].pct_change().dropna()

                # Bin returns
                bins = np.linspace(trades_df['returns'].min()*2, trades_df['returns'].max()*2, 500)
                trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

                # Calculate entropy for each window length
                for w in window_lengths:
                    trades_df[f'Entropy_{w}'] = trades_df['returns_binned'].rolling(window=w).apply(
                        lambda x: calculate_price_entropy(x.dropna()), raw=False
                    )

                    # Store the average entropy for this trial
                    avg_entropy_per_window[w].append(trades_df[f'Entropy_{w}'].mean())

            except FileNotFoundError:
                print(f"File {trades_file} not found, skipping...")
                continue

    # Compute the overall average entropy for each window length
    for w in window_lengths:
        entropy_results[n].append(np.nanmean(avg_entropy_per_window[w]))  # Handle NaNs safely

# Plot the results
plt.figure(figsize=(10, 6))
for n in num_traders:
    plt.plot(window_lengths, entropy_results[n], marker='x', label=f'{n} Traders', ls = ' ')

plt.xlabel('Rolling Window Length (Trades)')
plt.ylabel('Average Entropy')
plt.title('Effect of Window Length on Average Entropy')
plt.legend(title="Number of Traders")
plt.grid()
plt.show()
