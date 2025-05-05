import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BSE import market_session
import random

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})

def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0

def run_bse_and_get_trades(trial_id, n_traders, order_sched, start_time, end_time):
    traders_spec = {
        'sellers': [('GVWY', n_traders)],
        'buyers': [('GVWY', n_traders)]
    }

    market_session(
        trial_id, start_time, end_time, traders_spec, order_sched,
        {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
         'dump_avgbals': False, 'dump_tape': True}, False
    )

    trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
    trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
    trades_df = trades_df.sort_values('Time').set_index('Time')

    start_time = trades_df.index.min()
    trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()

    trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1)).dropna()
    trades_df['Time_seconds'] = trades_df.index.total_seconds()

    return trades_df

def compute_rolling_entropy(trades_df, window_length, bin_size=2000):
    time_window_length = f'{window_length}s'
    window_seconds = pd.to_timedelta(time_window_length).total_seconds()

    n_bins = bin_size
    bins = np.linspace(trades_df['returns'].min()*1.1, trades_df['returns'].max()*1.1, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    rolling_entropy = trades_df['returns_binned'].rolling(
        time_window_length, closed='both'
    ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

    valid_mask = trades_df['Elapsed'] >= window_seconds
    entropy = rolling_entropy.where(valid_mask)

    return entropy.mean() if not entropy.dropna().empty else np.nan

def main():
    # Market config
    start_time = 0
    end_time = 300
    range1 = (100, 150)
    range2 = (80, 130)
    order_interval = 1
    n_traders = 50
    trader_type = 'ZIP'

    supply_schedule = [
        {'from': 0, 'to': 150, 'ranges': [range1], 'stepmode': 'jittered'},
        {'from': 150, 'to': 300, 'ranges': [range2], 'stepmode': 'jittered'}
    ]
    order_sched = {
        'sup': supply_schedule,
        'dem': supply_schedule,
        'interval': order_interval,
        'timemode': 'drip-fixed'
    }

    window_lengths = [5, 45, 150]
    bin_size = 20000
    trial_id = f'{n_traders}_{trader_type}B_{trader_type}S_run0_{random.randint(1000,9999)}'

    trades_df = run_bse_and_get_trades(trial_id, n_traders, order_sched, start_time, end_time)

    # Prepare bins once
    bins = np.linspace(trades_df['returns'].min() * 1.1, trades_df['returns'].max() * 1.1, bin_size)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    # Compute entropy series for each window length
    entropy_series = {}
    for w in window_lengths:
        time_window_length = f'{w}s'
        window_seconds = pd.to_timedelta(time_window_length).total_seconds()

        rolling_entropy = trades_df['returns_binned'].rolling(
            time_window_length, closed='both'
        ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

        valid_mask = trades_df['Elapsed'] >= window_seconds
        entropy_series[w] = rolling_entropy.where(valid_mask)

    # Plotting: 1x3 layout, same price data in all, different entropy
    fig, axes = plt.subplots(1, 3, figsize=(25, 6), sharex=True)

    for idx, w in enumerate(window_lengths):
        ax = axes[idx]
        ax.plot(trades_df.index.total_seconds(), trades_df['Price'], color='black', ls = ' ', label='Price', marker = 'x', markersize = 2)
        ax.set_ylabel("Price", color='black')
        ax.tick_params(axis='y', labelcolor='black')

        ax2 = ax.twinx()
        ax2.plot(trades_df.index.total_seconds(), entropy_series[w], color='red', ls = '--', label=f'Entropy ({w}s)')
        ax2.set_ylabel("Entropy (bits)")
        ax2.tick_params(axis='y')

        ax.set_title(f'Window: {w}s')
        ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f'{trader_type}_{n_traders}_price_entropy_subplots.png')
    plt.show()


main()