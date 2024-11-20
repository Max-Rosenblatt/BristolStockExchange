import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv
import pandas as pd

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 1800
chart1_range = (10, 100)
order_interval = 1

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-poisson'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True, 'dump_tape': True}

# Define trader types and simulation setup
num_traders = 5
buyer_types = ['ZIP', 'ZIC', 'GVWY', 'SHVR', 'SNPR']
seller_type = 'ZIC'

# Loop over each buyer type
for buyer_type in buyer_types:
    trial_id = f'Test_{num_traders}_traders_{buyer_type}'

    # Specify sellers and buyers
    sellers_spec = [(seller_type, num_traders)]
    buyers_spec = [(buyer_type, num_traders)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # Run the market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

    # Load Level 2 (LOB) data
    l2_data = trial_id + '_LOB_frames.csv'
    time_price_pairs_bids = []
    time_price_pairs_ask = []
    best_prices = []

    with open(l2_data, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[0])
            bids = eval(row[1])
            asks = eval(row[3])
            bid_best = row[2]
            ask_best = row[4]

            # Append bid prices
            for bid in bids:
                price = bid[0]
                time_price_pairs_bids.append((time, price))

            # Append ask prices
            for ask in asks:
                price = ask[0]
                time_price_pairs_ask.append((time, price))

            # Process best bid/ask prices
            bid_best = float(bid_best) if bid_best != '' else None
            ask_best = float(ask_best) if ask_best != '' else None
            best_prices.append((time, bid_best, ask_best))

    times = [bp[0] for bp in best_prices]
    best_bids = [bp[1] for bp in best_prices]
    best_asks = [bp[2] for bp in best_prices]

    # Load trade prices
    prices_fname = trial_id + '_tape.csv'
    trade_times = []
    trade_prices = []
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[1])
            price = float(row[2])
            trade_times.append(time)
            trade_prices.append(price)

    # Calculate volatility
    df = pd.read_csv(prices_fname, names=['Type', 'Time', 'Price'])
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Return'] = df['Price'].pct_change()
    window_size = 5  # Adjust as needed
    df['Realized_Variance'] = df['Return'].rolling(window=window_size).var()
    df['Volatility'] = np.sqrt(df['Realized_Variance'])

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Bid/Ask Prices and Trades
    axs[0].plot(trade_times, trade_prices, 'x', color='black', label='Trades', markersize=12)
    #axs[0].plot(times, best_bids, label='Highest Bid Price', color='blue', marker='.', linestyle='')
    #axs[0].plot(times, best_asks, label='Lowest Ask Price', color='darkseagreen', marker='.', linestyle='')
    axs[0].set_ylabel('Prices (£)')
    axs[0].set_title(f"'{buyer_type}' Bidders vs. '{seller_type}' Sellers: Prices")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Volatility
    axs[1].plot(df['Time'], df['Volatility'], label='Volatility', color='purple')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Volatility')
    axs[1].set_title(f"Market Volatility Over Time for '{buyer_type}' Buyers")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and save the figure
    plt.show()
