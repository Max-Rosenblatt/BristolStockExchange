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
range1 = (100, 160)
range2 = (70, 130)
order_interval = 10

supply_schedule = [
    {'from': 0, 'to': 300, 'ranges': [range1], 'stepmode': 'jittered'},
    {'from': 300, 'to': 600, 'ranges': [range2], 'stepmode': 'jittered'}
]
demand_schedule = supply_schedule
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-fixed'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}

# Define buyer/seller types (Fixed)
buyer_types = ['ZIP', 'ZIC', 'SHVR', 'GVWY']
seller_types = ['ZIP', 'ZIC', 'SHVR', 'GVWY']

# Define number of traders to iterate over
num_trader_values = [10, 20, 30, 40, 50, 60, 80, 100]
average_entropies = []
for seller_type in seller_types:
    for buyer_type in buyer_types:
        for num_traders in num_trader_values:
            trial_id = f'{num_traders}_{buyer_type}B_{seller_type}S'

            # Specify sellers and buyers
            sellers_spec = [(seller_type, num_traders)]
            buyers_spec = [(buyer_type, num_traders)]
            traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

            # Fixed parameters
            n_bins = 500
            window_length = 50  # Fixed rolling window

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
            bins = np.linspace(trades_df['returns'].min()*2, trades_df['returns'].max()*2, n_bins)
            trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
            trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=window_length).apply(
                lambda x: calculate_price_entropy(x.dropna()), raw=False)

            # Compute average entropy (excluding NaN values)
            avg_entropy = trades_df['Entropy'].mean()
            average_entropies.append(avg_entropy)
            print(num_traders, buyer_type, avg_entropy)

