import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 600
chart1_range = (1, 100)
order_interval = 10

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True, 'dump_tape': True}

# Loop over different numbers of traders
num_traders = 50

# Define buyer types
buyer_types = ['ZIP']
seller_type = 'ZIC'

# Loop over each buyer type
for buyer_type in buyer_types:
    trial_id = f'Test_{num_traders}_traders_{buyer_type}'

    # Specify sellers and buyers based on the current number of traders
    sellers_spec = [(seller_type, num_traders)]
    buyers_spec = [(buyer_type, num_traders)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # Run the market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

    l2_data = trial_id + '_LOB_frames.csv'
    time_price_pairs_bids = []  # Initialize a list to hold (time, price) pairs for bids
    time_price_pairs_ask = []  # Initialize a list for asks
    best_prices = []

    with open(l2_data, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[0])  # Get time as a float
            bids = eval(row[1])  # Evaluate the string to a list of tuples for bids
            asks = eval(row[3])  # Evaluate the string to a list of tuples for asks
            bid_best = row[2]
            ask_best = row[4]

            bid_prices = [bid[0] for bid in bids]  # Extract the price from each bid tuple
            ask_prices = [ask[0] for ask in asks]  # Extract the price from each ask tuple


            # Append bid prices
            for bid in bids:
                bid_price = bid[0]  # Get the bid price (first element of the tuple)
                time_price_pairs_bids.append((time, bid_price))  # Add tuple of (time, bid price)

            # Append ask prices
            for ask in asks:
                ask_price = ask[0]  # Get the ask price (first element of the tuple)
                time_price_pairs_ask.append((time, ask_price))  # Add tuple of (time, ask price)




            if bid_best != '':
                bid_best = float(bid_best)
            else:
                bid_best = None  # or '' for an empty string

            if ask_best != '':
                ask_best = float(ask_best)
            else:
                ask_best = None  # or '' for an empty string

            best_prices.append((time, bid_best, ask_best))

    times = [bp[0] for bp in best_prices]
    best_bids = [bp[1] for bp in best_prices]
    best_asks = [bp[2] for bp in best_prices]

    # Convert to numpy arrays for easier plotting
    time_price_array_bids = np.array(time_price_pairs_bids)
    time_price_array_asks = np.array(time_price_pairs_ask)

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