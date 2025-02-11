import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session


# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


# Simulation parameters
start_time = 0
end_time = 60
chart1_range = (1, 100)
order_interval = 1

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

# Parameter ranges
n = 100
seller_types = ['ZIC', 'ZIP']  # Example seller types
bin_sizes = np.linspace(10, 5000, 25, dtype=int)
window_length = 10

results = []

for seller_type in seller_types:
    for binsize in bin_sizes:
        buyer_type = 'ZIP'
        trial_id = f'{n}_traders_{buyer_type}_{seller_type}_bins{binsize}_win{window_length}'

        sellers_spec = [(seller_type, n)]
        buyers_spec = [(buyer_type, n)]
        traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

        dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True,
                      'dump_tape': True}
        verbose = False

        # Run market session
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

        # Read in market data
        trades_file = f'{trial_id}_tape.csv'
        trades_df = pd.read_csv(trades_file)
        trades_df.columns = ['Type', 'Time', 'Price']
        trades_df['id'] = range(len(trades_df))
        trades_df.set_index('id', inplace=True)
        trades_df = trades_df.sort_values(by="Time")
        trades_df['returns'] = trades_df['Price'].pct_change().dropna()

        # Bin returns and calculate entropy
        bins = np.linspace(-10, 10, binsize)
        trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
        trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=window_length).apply(
            lambda x: calculate_price_entropy(x.dropna()))

        avg_entropy = trades_df['Entropy'].mean()
        results.append((n, seller_type, binsize, window_length, avg_entropy))
        print(
            f"Traders: {n}, Seller: {seller_type}, Bins: {binsize}, Window: {window_length}, Entropy: {avg_entropy}")

# Convert results to DataFrame and plot
results_df = pd.DataFrame(results, columns=['Num Traders', 'Seller Type', 'Bin Size', 'Window Length', 'Avg Entropy'])
plt.figure(figsize=(10, 5))
for seller in seller_types:
    subset = results_df[results_df['Seller Type'] == seller]
    plt.plot(subset['Bin Size'], subset['Avg Entropy'], marker='x', linestyle='', label=seller)
plt.xlabel("Bin Size")
plt.ylabel("Average Entropy")
plt.title("Market Entropy Analysis")
plt.legend()
plt.show()
