import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
from BSE import market_session

# Create plots folder if not exists
os.makedirs("plots", exist_ok=True)

def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0

def run_single_session(buyer_type, seller_type, trial_id, end_time=300, n_traders=50, window='45s'):
    # No shock
    range_constant = (100, 150)
    supply_schedule = [
        {'from': 0, 'to': end_time, 'ranges': [range_constant], 'stepmode': 'jittered'}
    ]
    demand_schedule = supply_schedule
    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': 20,
        'timemode': 'drip-fixed'
    }

    traders_spec = {
        'sellers': [(seller_type, n_traders)],
        'buyers': [(buyer_type, n_traders)]
    }

    market_session(trial_id, 0, end_time, traders_spec, order_sched,
                   {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                    'dump_avgbals': False, 'dump_tape': True}, False)

    try:
        trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
        trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
        trades_df = trades_df.sort_values('Time').set_index('Time')

        trades_df['Elapsed'] = (trades_df.index - trades_df.index.min()).total_seconds()
        trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1)).dropna()
        n_bins = 1000
        bins = np.linspace(trades_df['returns'].min() * 1.1, trades_df['returns'].max() * 1.1, n_bins)
        trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
        entropy_series = trades_df['returns_binned'].rolling(
            window, closed='both'
        ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

        avg_entropy = entropy_series.mean()

        # Plotting price over time
        trades_df['Time_seconds'] = trades_df.index.total_seconds()
        plt.figure(figsize=(10, 5))
        plt.plot(trades_df['Time_seconds'], trades_df['Price'], marker='x', linestyle=' ', color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Price')
        plt.title(f'Trade Price Over Time: {trial_id}')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in {trial_id}: {e}")
        avg_entropy = np.nan

    return avg_entropy

# Parameters
trader_types = ['ZIC', 'ZIP', 'SHVR', 'GVWY']
n_repeats = 5
results = []

combinations = list(itertools.product(trader_types, repeat=2))

print("Running all trader combinations...\n")
for idx, (buyer_type, seller_type) in enumerate(combinations, 1):
    print(f'Running combo {idx}/{len(combinations)}: {buyer_type}B / {seller_type}S')

    entropy_list = []
    for i in range(n_repeats):
        trial_id = f'{buyer_type}B_{seller_type}S_{i}'
        avg_entropy = run_single_session(buyer_type, seller_type, trial_id)
        entropy_list.append(avg_entropy)

    results.append({
        'Buyer': buyer_type,
        'Seller': seller_type,
        'AverageEntropy': np.nanmean(entropy_list)
    })

# Save CSV
df = pd.DataFrame(results)
df.to_csv('entropy_results_no_shock.csv', index=False)
print("\nAll simulations complete. Results saved to 'entropy_results_no_shock.csv'")
