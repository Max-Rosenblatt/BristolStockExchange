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
dump_flags = {
    'dump_blotters': False,
    'dump_lobs': True,
    'dump_strats': False,
    'dump_avgbals': False,
    'dump_tape': True
}

# Define the list of numbers of traders you want to test
num_traders_list = np.linspace(20,50,25, dtype = int)

# Fixed parameters for entropy calculation
n_bins = 2000
time_window = 30  # rolling window of 50 seconds

# Define shock time as 300 seconds
shock_time = pd.Timedelta(seconds=300)

# This list will store the entropy jump (post - pre) for each simulation
entropy_jumps = []

# Loop over each number of traders (using "ZIP" for both buyers and sellers)
for n in num_traders_list:
    
    trial_id = f'{n}_ZIPB_ZIPS'
    sellers_spec = [('ZIP', n)]
    buyers_spec = [('ZIP', n)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # Run the market simulation
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

    # Read the market data
    trades_file = f'{trial_id}_tape.csv'
    trades_df = pd.read_csv(trades_file)
    trades_df.columns = ['Type', 'Time', 'Price']
    trades_df['id'] = range(len(trades_df))
    trades_df['seller'] = 'ZIP'
    trades_df = trades_df.sort_values(by="Time")
    trades_df['returns'] = trades_df['Price'].pct_change().dropna()

    # Convert 'Time' (in seconds) to a timedelta for time-aware rolling and filtering
    trades_df['Time_dt'] = pd.to_timedelta(trades_df['Time'], unit='s')
    trades_df.set_index('Time_dt', inplace=True)

    # Bin the returns for entropy calculation
    bins = np.linspace(trades_df['returns'].min() * 2, trades_df['returns'].max() * 2, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    # Calculate the rolling entropy over a time-based window
    trades_df['Entropy'] = trades_df['returns_binned'].rolling(f'{time_window}s').apply(
        lambda x: calculate_price_entropy(x.dropna()), raw=False)

    # Reset the index so that Time_dt becomes a column again for filtering
    trades_df_reset = trades_df.reset_index()

    # Compute average entropy in the 1-minute interval before and after the shock (300 sec)
    pre_shock = trades_df_reset[
        (trades_df_reset['Time_dt'] >= shock_time - pd.Timedelta(minutes=1)) &
        (trades_df_reset['Time_dt'] < shock_time)
        ]
    post_shock = trades_df_reset[
        (trades_df_reset['Time_dt'] >= shock_time) &
        (trades_df_reset['Time_dt'] < shock_time + pd.Timedelta(minutes=1))
        ]

    avg_pre = pre_shock['Entropy'].mean()
    avg_post = post_shock['Entropy'].mean()
    entropy_jump = avg_post - avg_pre
    print(
        f"Traders: {n} - Pre-shock avg entropy: {avg_pre:.4f}, Post-shock avg entropy: {avg_post:.4f}, Jump: {entropy_jump:.4f}")

    entropy_jumps.append(entropy_jump)

# Plot the entropy jump vs. number of traders
plt.figure(figsize=(8, 6))
plt.plot(num_traders_list, entropy_jumps, marker='o', linestyle=' ')
plt.xlabel("Number of Traders")
plt.ylabel("Entropy Jump (Post - Pre)")
plt.title("Entropy Jump vs. Number of Traders")
plt.grid(True)
plt.show()
