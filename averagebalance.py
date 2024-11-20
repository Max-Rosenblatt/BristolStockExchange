import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
import numpy as np
import csv

from BSE import market_session

# Simulation parameters
start_time = 0
end_time = 600
chart1_range = (1, 100)
order_interval = 1

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

verbose = False
dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True, 'dump_tape': True}

# Loop over different numbers of traders
num_traders = 50

# Define buyer types
buyer_types = ['ZIP', 'GVWY', 'SHVR', 'SNPR']
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

    data = trial_id + '_avg_balance.csv'

    time = []
    profit = []
    with open(data, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time.append(float(row[1]))  # Get time as a float
            profit.append(float(row[11]))  # Evaluate the string to a list of tuples for bids

    # Plot bid and ask prices against time on the same axes
    plt.figure(figsize=(12, 6))
    plt.plot(time, profit, 'x', color='black', label='Trades', markersize=12)
    plt.xlabel('Time (s)')
    plt.ylabel(f'Profit for {buyer_type} Buyer (£)')
    plt.xlim(0,600)
    plt.legend()
    plt.grid(True)
    plt.show()

