import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 200
chart1_range = (10, 100)
order_interval = 5

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-poisson'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}

# Loop over different numbers of traders
num_traders = 5
trial_id = f'Test_{num_traders}_traders'

# Specify sellers and buyers based on the current number of traders
sellers_spec = [('ZIP', num_traders)]
buyers_spec = [('ZIP', 5)]
traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

# Run the market session
market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

l2_data = trial_id + '_LOB_frames.csv'
time_price_pairs_bids = []  # Initialize a list to hold (time, price) pairs for bids
time_price_pairs_ask = []  # Initialize a list for asks

with open(l2_data, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        time = float(row[0])  # Get time as a float
        bids = eval(row[1])  # Evaluate the string to a list of tuples for bids
        asks = eval(row[2])  # Evaluate the string to a list of tuples for asks

        # Append bid prices
        for bid in bids:
            price = bid[0]  # Get the bid price (first element of the tuple)
            time_price_pairs_bids.append((time, price))  # Add tuple of (time, bid price)

        # Append ask prices
        for ask in asks:
            price = ask[0]  # Get the ask price (first element of the tuple)
            time_price_pairs_ask.append((time, price))  # Add tuple of (time, ask price)

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



# Plot bid and ask prices against time on the same axes
plt.figure(figsize=(12, 6))
plt.plot(time_price_array_bids[:, 0], time_price_array_bids[:, 1], marker='.', linestyle='', color='green',
         label='Bid Prices')
plt.plot(time_price_array_asks[:, 0], time_price_array_asks[:, 1], marker='.', linestyle='', color='red',
         label='Ask Prices')
plt.plot(trade_times, trade_prices, 'x', color='blue', markersize=20);
plt.xlabel('Time (s)')
plt.ylabel('Prices (£)')
plt.title('Bid and Ask Prices Over Time')
plt.legend()
plt.grid(True)
plt.show()
