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
range2 = (80, 130)
order_interval = 1
n_traders = 50
trader_type = 'ZIP'
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
    {'from': 150, 'to': 300, 'ranges': [range2], 'stepmode': 'jittered'}
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
trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')  # Convert to timedelta
trades_df = trades_df.sort_values('Time').set_index('Time')

start_time = trades_df.index.min()
trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()
window_seconds = pd.to_timedelta(time_window_length).total_seconds()

# Calculate returns and bin data
trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1)).dropna()
trades_df['Volatility'] = trades_df['returns'].rolling(window=time_window_length, closed='both').std()
n_bins = 20000
bins = np.linspace(trades_df['returns'].min()*1.5, trades_df['returns'].max()*1.5, n_bins)
trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
rolling_entropy = trades_df['returns_binned'].rolling(
    time_window_length, closed='both'
).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

trades_df['Entropy'] = np.nan  # Initialize with NaN

# Filter to only include points where elapsed time >= window_seconds
valid_mask = trades_df['Elapsed'] >= window_seconds
filtered_rolling_entropy = rolling_entropy.where(valid_mask)
trades_df['Volatility'] = trades_df['Volatility'].where(valid_mask)

# Store the result back - this will keep NaN for early times
trades_df['Entropy'] = filtered_rolling_entropy

# Convert index back to seconds for plotting
trades_df['Time_seconds'] = trades_df.index.total_seconds()

pre_shock = trades_df[(trades_df['Time_seconds'] <= shock_time)]
post_shock = trades_df[(trades_df['Time_seconds'] >= shock_time) & (trades_df['Time_seconds'] < shock_time+30)]

pre_avg = pre_shock['Entropy'].mean() if not pre_shock['Entropy'].dropna().empty else np.nan
post_avg = post_shock['Entropy'].mean() if not post_shock['Entropy'].dropna().empty else np.nan

trades_df.to_csv(f'{trial_id}_stats.csv')

print(f"Pre-shock Entropy = {pre_avg:.4f}, Post-shock Entropy = {post_avg:.4f}")

# Create two vertically stacked subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# --- Bottom subplot: Volatility ---
ax1.plot(trades_df['Time_seconds'], trades_df['Price'], linestyle=' ', color='black', marker = '.', markersize = 2)
ax1.set_ylabel('Trade Price')
yticks = np.arange(80, 150, 20)
ax1.set_yticks(yticks)


# --- Top subplot: Closing Price and Entropy ---
ax2.set_ylabel('Volatility')
ax2.plot(trades_df['Time_seconds'], trades_df['Volatility'], linestyle='-', color='blue', label='Volatility')
ax2b = ax2.twinx()
ax2b.set_ylabel('Entropy / Bits')
ax2b.plot(trades_df['Time_seconds'], trades_df['Entropy'], linestyle='-', color='red', label='Entropy')
ax2.set_xlabel('Time/s')

plt.tight_layout()
plt.savefig('price_entropy_volatility_ZIP.png', dpi=300)
plt.show()
