import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})


def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0

# Simulation parameters
start_time = 0
end_time = 300
range1 = (100, 150)
range2 = (90, 140)
order_interval = 1
n_traders = 50
trader_type = 'GVWY'
window = 45
time_window_length = f'{window}s'

shock_time = 150
pre_start = shock_time - window
pre_end = shock_time
post_start = shock_time
post_end = shock_time + window

# Configure market session
supply_schedule = [
    {'from': 0, 'to': 150, 'ranges': [range1], 'stepmode': 'jittered'},
    {'from': 150, 'to': 300, 'ranges': [range1], 'stepmode': 'jittered'}
]
demand_schedule = supply_schedule
order_sched = {
    'sup': supply_schedule,
    'dem': demand_schedule,
    'interval': order_interval,
    'timemode': 'drip-fixed'
}

trial_id = f'{n_traders}_{trader_type}B_{trader_type}S'
traders_spec = {
    'sellers': [(trader_type, n_traders)],
    'buyers': [(trader_type, n_traders)]
}

# Run market session
market_session(trial_id, start_time, end_time, traders_spec, order_sched,
               {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False,
                'dump_avgbals': False, 'dump_tape': True}, False)

# Process results
trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
trades_df = trades_df.sort_values('Time').set_index('Time')

start_time = trades_df.index.min()
trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()

trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1))
trades_df['returns'] = trades_df['returns'].dropna()

n_bins = 2000
bins = np.linspace(trades_df['returns'].min() * 1.5, trades_df['returns'].max() * 1.5, n_bins)
trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

trades_df['Entropy'] = np.nan
valid_idx = trades_df[trades_df['Elapsed'] >= pd.to_timedelta(time_window_length).total_seconds()].index

# Perform rolling on the full series first, then extract valid values
rolling_entropy = trades_df['returns_binned'].rolling(
    time_window_length, closed='both'
).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

# Assign the computed entropy values to the DataFrame
trades_df['Entropy'] = rolling_entropy

trades_df.loc[trades_df['Elapsed'] < pd.to_timedelta(time_window_length).total_seconds(), 'Entropy'] = np.nan

# Add a simple column for elapsed time in seconds:
trades_df['Time_seconds'] = trades_df.index.total_seconds()


pre_shock = trades_df[(trades_df['Time_seconds'] >= pre_start) & (trades_df['Time_seconds'] < pre_end)]
post_shock = trades_df[(trades_df['Time_seconds'] >= post_start) & (trades_df['Time_seconds'] < post_end)]

pre_avg = pre_shock['Entropy'].mean() if not pre_shock['Entropy'].dropna().empty else np.nan
post_avg = post_shock['Entropy'].mean() if not post_shock['Entropy'].dropna().empty else np.nan

trades_df.to_csv(f'{trial_id}_stats.csv')

print(f"Pre-shock Entropy = {pre_avg:.4f}, Post-shock Entropy = {post_avg:.4f}")

# Create visualizations
# Create visualization with dual axes
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot Price on primary axis (left)
ax1.scatter(trades_df['Time_seconds'], trades_df['Price'], marker='x', color='black', alpha=0.7, label='Trades')
ax1.set_xlabel("Time/s")
ax1.set_ylabel("Trade Price")
ax1.tick_params(axis='y')
ax1.set_ylim(100,150)

# Create secondary axis (right) for Entropy
ax2 = ax1.twinx()
ax2.plot(trades_df['Time_seconds'], trades_df['Entropy'], ls='--', color='red', label='Entropy', lw=3)
ax2.set_ylabel("Entropy/bits")
ax2.tick_params(axis='y')
ax2.set_ylim(0,10)
plt.tight_layout()

plt.savefig('gvwy_vs_gvwy.png', dpi = 300)

plt.show()

