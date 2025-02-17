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
end_time = 600
range1 = (10,100)
range2 = (100,190)
order_interval = 10

supply_schedule = [ {'from':0, 'to':300, 'ranges':[range1], 'stepmode':'jittered'},
                    {'from':300, 'to':360, 'ranges':[range2], 'stepmode':'jittered'},
                    {'from':360, 'to':600, 'ranges':[range1], 'stepmode':'jittered'}]
demand_schedule = supply_schedule
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-fixed'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}

# Loop over different numbers of traders
num_traders = 50

# Define buyer types
buyer_type = 'ZIP'
seller_type = 'ZIP'

# Loop over each buyer type
trial_id = f'{num_traders}_{buyer_type}_buyers_vs_{seller_type}_sellers'

# Specify sellers and buyers based on the current number of traders
sellers_spec = [(seller_type, num_traders)]
buyers_spec = [(buyer_type, num_traders)]
traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

# Fixed parameters
n_bins = 20000
window_length = 200  # Fixed rolling window

# Run market session
market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

# Read in market data
trades_file = f'{trial_id}_tape.csv'
trades_df = pd.read_csv(trades_file)
trades_df.columns = ['Type', 'Time', 'Price']
trades_df['id'] = range(len(trades_df))
trades_df['seller'] = seller_type
trades_df.set_index('id', inplace=True)
trades_df = trades_df.sort_values(by="Time")
trades_df['returns'] = trades_df['Price'].pct_change().dropna()

# Bin returns and calculate entropy
bins = np.linspace(-1000, 1000, n_bins)
trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=window_length).apply(
    lambda x: calculate_price_entropy(x.dropna()), raw=False)

# Append to list

# Create subplots for Price and Entropy
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Scatter plot for trade prices
axes[0].scatter(trades_df['Time'], trades_df['Price'], marker='x', label=f'{seller_type}', color='black')
axes[0].set_ylabel("Trade Price")
axes[0].set_title(f"Trade Prices Over Time ({seller_type})")
axes[0].legend()
axes[0].grid()

# Scatter plot for entropy
axes[1].scatter(trades_df['Time'], trades_df['Entropy'], marker='x', label=f'Entropy ({seller_type})', color='blue')
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Entropy")
axes[1].set_title(f"Price Entropy Over Time ({seller_type})")
axes[1].legend()
axes[1].grid()

axes[0].set_xlim(start_time, end_time)
axes[1].set_xlim(start_time, end_time)

plt.tight_layout()
plt.show()

