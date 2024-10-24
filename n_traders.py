import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv

import random

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 60 * 10
chart1_range = (10, 100)
order_interval = 5

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}

# Colors for plotting each simulation's data
colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

# Loop over different numbers of traders
for i, num_traders in enumerate(range(10, 51, 5)):  # Example: incrementing traders from 10 to 100 in steps of 10
    trial_id = f'Test_{num_traders}_traders'

    # Specify sellers and buyers based on the current number of traders
    sellers_spec = [('ZIP', num_traders)]
    buyers_spec = [('ZIP', 25)]
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
    plt.plot(x, y, 'o', color=colors[i % len(colors)], label=f'{num_traders} Traders')

"""
# Finalize the plot with labels and legend
plt.xlabel('Time/s')
plt.ylabel('Price/£')
plt.legend(loc='best', fontsize='small')
plt.title('Effect of Number of Traders on Market Prices')
plt.show()
"""
