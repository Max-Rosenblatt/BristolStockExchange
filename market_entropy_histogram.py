import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
from BSE import market_session

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Parameters
seller_type = 'GVWY'  # Fixed seller type
buyer_types = ['ZIC', 'ZIP', 'SHVR', 'GVWY']  # Varying buyer types
n_repeats = 100
end_time = 300
n_traders = 50
data_file = output_dir / "returns_data.pkl"
force_rerun = False  # Set to True to force new simulations


def run_session_and_get_returns(buyer_type, seller_type, trial_id, end_time=300):
    """Run a market session and return the log returns series"""
    range_constant = (100, 150)
    supply_schedule = [{'from': 0, 'to': end_time, 'ranges': [range_constant], 'stepmode': 'jittered'}]
    demand_schedule = supply_schedule

    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': 1,
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
        trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1))
        return trades_df['returns'].dropna()
    except Exception as e:
        print(f"Error in {trial_id}: {e}")
        return pd.Series(dtype=float)


# Check for existing data file:
if not force_rerun and data_file.exists():
    print(f"Loading existing data from {data_file}")
    try:
        with open(data_file, 'rb') as f:
            returns_data = pickle.load(f)
        print("Successfully loaded saved data")
    except Exception as e:
        print(f"Error loading saved data: {e}. Will run new simulations.")
        returns_data = {buyer: [] for buyer in buyer_types}
else:
    print("Data file not found or force_rerun is set. Running new simulations...")
    returns_data = {buyer: [] for buyer in buyer_types}
    for buyer_type in buyer_types:
        print(f"\nRunning buyer type: {buyer_type}")
        for i in range(n_repeats):
            trial_id = f'GVWY_S_{buyer_type}B_{i}'
            tape_file = Path(f'{trial_id}_tape.csv')
            if not force_rerun and tape_file.exists():
                print(f"  Trial {i + 1}: Loading existing data from tape file")
                returns = run_session_and_get_returns(buyer_type, seller_type, trial_id, end_time)
            else:
                print(f"  Trial {i + 1}: Running new simulation")
                returns = run_session_and_get_returns(buyer_type, seller_type, trial_id, end_time)
            returns_data[buyer_type].extend(returns.values)
            print(f"  Collected {len(returns)} returns (total: {len(returns_data[buyer_type])})")

    # Save the collected data to the file since it doesn't exist yet
    with open(data_file, 'wb') as f:
        pickle.dump(returns_data, f)
    print(f"\nSaved returns data to {data_file}")

# Verify we have data for all buyer types
for buyer in buyer_types:
    if not returns_data.get(buyer):
        print(f"Warning: No data available for buyer type {buyer}")

# Create comparative histogram plot with improved styling
plt.figure(figsize=(14, 7))
# Map colors for each buyer type: ZIC: green, ZIP: red, SHVR: yellow, GVWY: blue.
colors = ['green', 'red', 'orange', 'blue']
hatches = ['/', '\\', '|', '-']

for i, (buyer_type, color, hatch) in enumerate(zip(buyer_types, colors, hatches)):
    returns = returns_data.get(buyer_type, [])
    if returns:  # Only plot if we have data
        n, bins, patches = plt.hist(returns, bins=25, density=False,
                                    alpha=0.3, color=color, label=f'{buyer_type}',
                                    edgecolor='black', linewidth=0.5)

        # Apply hatching to each bar for distinction
        for patch in patches:
            patch.set_hatch(hatch)
            patch.set_linewidth(0.8)

plt.xlabel('Log Return')
plt.ylabel('Count')
# After plotting, retrieve current legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Zip handles and labels, then sort by the label (alphabetically)
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
sorted_labels, sorted_handles = zip(*sorted_handles_labels)

# Create the legend with the sorted handles and labels
plt.legend(sorted_handles, sorted_labels, fontsize=20)



plt.tight_layout()

plot_path = output_dir / "comparative_returns_distribution.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved comparative histogram to {plot_path}")
plt.show()