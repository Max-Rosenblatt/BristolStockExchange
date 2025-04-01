import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session

def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0

# Simulation parameters
start_time = 0
end_time = 300
range1 = (100, 160)
range2 = (70, 130)
order_interval = 1
n_traders = 50
trader_type = 'ZIP'
time_window_length = '5s'

# Configure market session
supply_schedule = [
    {'from': 0, 'to': 150, 'ranges': [range1], 'stepmode': 'fixed'},
    {'from': 150, 'to': 300, 'ranges': [range2], 'stepmode': 'fixed'}
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

# Calculate returns and bin data
trades_df['returns'] = trades_df['Price'].pct_change().dropna()
n_bins = 2000
bins = np.linspace(trades_df['returns'].min()*1.5, trades_df['returns'].max()*1.5, n_bins)
trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

# Calculate rolling entropy based on time window
trades_df['Entropy'] = trades_df['returns_binned'].rolling(
    time_window_length,
    closed='both'
).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

# Convert index back to seconds for plotting
trades_df['Time_seconds'] = trades_df.index.total_seconds()
trades_df.to_csv(f'{trial_id}_stats.csv')

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot price data on primary y-axis
scatter = ax1.scatter(trades_df['Time_seconds'], trades_df['Price'],
                     marker='x', color='black', alpha=0.7, label='Trade Price')
ax1.set_ylabel("Trade Price", color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xlabel("Time (seconds)")
ax1.set_title(f"Trade Prices and Entropy Over Time ({trader_type} Traders)")
ax1.grid(True)

# Create a secondary y-axis for entropy
ax2 = ax1.twinx()
line, = ax2.plot(trades_df['Time_seconds'], trades_df['Entropy'],
                linestyle=':', color='blue', alpha=0.7, label='Entropy')
ax2.set_ylabel("Entropy", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Combine legends from both axes
lines = [scatter, line]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.show()

plt.hist(trades_df['returns'] * 100, bins=50, alpha=0.5)
plt.xlabel('Price Change between Adjacent Trades (%)')
plt.ylabel('Frequency')
plt.title(f'Price Change Distribution: {trader_type} Traders ({n_traders})')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()