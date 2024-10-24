import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv

import random

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 60 * 5
chart1_range = (10, 100)
order_interval = 5
num_traders = 50  # Keeping the number of traders constant

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}

# List of trader types to test
trader_types = ['ZIP', 'ZIC', 'GVWY', 'SHVR', 'SNPR']

# Colors for plotting each simulation's data
colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Loop over different trader types
for i, trader_type in enumerate(trader_types):
    trial_id = f'Test_{trader_type}_traders'

    # Specify sellers and buyers using the current trader type
    sellers_spec = [('ZIP', num_traders)]
    buyers_spec = [(trader_type, num_traders)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    # Run the market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

    # Load the generated data
    prices_fname = trial_id + '_tape.csv'
    x = np.empty(0)
    y = np.empty(0)
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[1])
            price = float(row[2])
            x = np.append(x, time)
            y = np.append(y, price)

    # Plot the data on the same graph using a different color
    plt.plot(x, y, 'o', color=colors[i % len(colors)], label=f'{trader_type} Traders')

# Finalize the plot with labels and legend
plt.xlabel('Time/s')
plt.ylabel('Price/£')
plt.legend(loc='best', fontsize='small')
plt.title('Effect of Trader Types on Market Prices')
plt.show()
