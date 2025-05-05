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
order_interval = 10
n_traders = 50
trader_type = 'ZIP'
time_window_length = '45s'

shock_time = 150
shock_window = 60  # +/- seconds around the shock

pre_start = shock_time - shock_window
pre_end = shock_time
post_start = shock_time
post_end = shock_time + shock_window

# CSV file for summary
summary_filename = 'entropy_summary.csv'
pd.DataFrame(columns=['Run', 'PreEntropy', 'PostEntropy']).to_csv(summary_filename, index=False)

for run in range(1, 101):
    trial_id = f'run{run}_{n_traders}_{trader_type}B_{trader_type}S'
    traders_spec = {
        'sellers': [(trader_type, n_traders)],
        'buyers': [(trader_type, n_traders)]
    }

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

    # Run market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched,
                   {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False,
                    'dump_avgbals': False, 'dump_tape': True}, False)

    # Process results
    trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
    trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
    trades_df = trades_df.sort_values('Time').set_index('Time')

    session_start = trades_df.index.min()
    trades_df['Elapsed'] = (trades_df.index - session_start).total_seconds()
    window_seconds = pd.to_timedelta(time_window_length).total_seconds()

    trades_df['returns'] = trades_df['Price'].pct_change().dropna()
    n_bins = 2000
    bins = np.linspace(trades_df['returns'].min()*1.5, trades_df['returns'].max()*1.5, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    trades_df['Entropy'] = np.nan
    valid_idx = trades_df[trades_df['Elapsed'] >= window_seconds].index

    rolling_entropy = trades_df.loc[valid_idx, 'returns_binned'].rolling(
        time_window_length, closed='both'
    ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

    trades_df.loc[rolling_entropy.index, 'Entropy'] = rolling_entropy
    trades_df['Time_seconds'] = trades_df.index.total_seconds()

    pre_shock = trades_df[(trades_df['Time_seconds'] >= pre_start) & (trades_df['Time_seconds'] < pre_end)]
    post_shock = trades_df[(trades_df['Time_seconds'] >= post_start) & (trades_df['Time_seconds'] < post_end)]

    pre_avg = pre_shock['Entropy'].mean() if not pre_shock['Entropy'].dropna().empty else np.nan
    post_avg = post_shock['Entropy'].mean() if not post_shock['Entropy'].dropna().empty else np.nan

    trades_df.to_csv(f'{trial_id}_stats.csv')

    print(f"Run {run} | Pre-shock Entropy = {pre_avg:.4f}, Post-shock Entropy = {post_avg:.4f}")

    # Plotting (unchanged)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].scatter(trades_df['Time_seconds'], trades_df['Price'], marker='x', color='black')
    axes[0].set_ylabel("Trade Price")
    axes[0].set_title(f"Trade Prices Over Time ({trader_type} Traders)")
    axes[0].grid()

    axes[1].scatter(trades_df['Time_seconds'], trades_df['Entropy'], marker='x', color='blue')
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Entropy")
    axes[1].set_title(f"Price Entropy Over {time_window_length} Window ({trader_type} Traders)")
    axes[1].plot([pre_start, pre_end], [pre_avg, pre_avg], color='red', linestyle='--', label='Pre-shock Avg Entropy')
    axes[1].plot([post_start, post_end], [post_avg, post_avg], color='green', linestyle='--', label='Post-shock Avg Entropy')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

    # Append to summary CSV
    with open(summary_filename, 'a') as f:
        f.write(f"{run},{pre_avg},{post_avg}\n")
