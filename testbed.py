import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

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
num_traders = 5

# Define buyer types
buyer_type = 'ZIP'
seller_type = 'ZIC'

trial_id = f'Test_{num_traders}_traders_{buyer_type}'

# Specify sellers and buyers based on the current number of traders
sellers_spec = [(seller_type, num_traders)]
buyers_spec = [(buyer_type, num_traders)]
traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

# Run the market session
market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

# Plot setup
fig, ax = plt.subplots()
fig.suptitle(f"Order Book Depth Chart", fontsize=16)
ax.set_xlabel("Price")
ax.set_ylabel("Order Count")
bid_line, = ax.plot([], [], color="blue", label="Bids", marker='x', ls='')  # Bids on the left
ask_line, = ax.plot([], [], color="red", label="Asks", marker='x', ls='')  # Asks on the right
trade_marker, = ax.plot([], [], 'go', label="Last Trade")  # Marker for trade price
ax.legend()

# Function to update plot dynamically
def update(frame):
    time, bids, asks = order_book_data[frame]

    # Extract bid and ask prices
    bid_prices = [bid[0] for bid in bids]
    ask_prices = [ask[0] for ask in asks]

    # Update bid and ask lines (just plot prices, no cumulative size)
    bid_line.set_data([1] * len(bid_prices), bid_prices)  # Place all bids at y=1
    ask_line.set_data([-1] * len(bid_prices), ask_prices)  # Place all asks at y=1

    # Update trade marker
    trade_price = trade_data.get(time)
    if trade_price:
        trade_marker.set_data([trade_price], [0])  # Place marker on x-axis

    ax.set_title(f"Order Book Depth Chart: t = {time:.5f}", fontsize=16)
    return bid_line, ask_line, trade_marker

# Load market data
l2_data = trial_id + '_LOB_frames.csv'
prices_fname = trial_id + '_tape.csv'

# Parse trade data
trade_data = {}
with open(prices_fname, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        time = float(row[1])
        price = float(row[2])
        trade_data[time] = price

# Parse order book data
order_book_data = []
with open(l2_data, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        time = float(row[0])
        bids = eval(row[1])
        asks = eval(row[3])
        order_book_data.append((time, bids, asks))

# Create animation
ani = FuncAnimation(fig, update, frames=len(order_book_data), interval=1, blit=True)

# Show the plot
plt.show()
