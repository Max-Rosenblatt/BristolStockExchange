import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import time  # For real-time delay simulation

from BSE import Exchange, Trader_Giveaway, Trader_ZIC

# Initialize market parameters
start_time = 0
end_time = 60 * 10
chart1_range = (80, 320)

supply_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]

order_interval = 60
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'periodic'}

sellers_spec = [('ZIP', 2)]
buyers_spec = [('ZIP', 2)]
traders_spec = {'sellers': sellers_spec,
                'buyers': buyers_spec}

# Create the market session manually
exchange = Exchange()

# Generate traders
traders = {}
for i in range(traders_spec['sellers'][0][1]):
    trader_id = f"S{str(i)}"
    traders[trader_id] = Trader_ZIC(trader_id, 'seller', 0.00)

for i in range(traders_spec['buyers'][0][1]):
    trader_id = f"B{str(i)}"
    traders[trader_id] = Trader_ZIC(trader_id, 'buyer', 0.00)

# Custom loop to simulate market and print updates in real-time
current_time = start_time
while current_time < end_time:
    # Each time step, generate a new order for each trader and process the market
    for trader in traders.values():
        order = trader.get_order(current_time, order_sched)
        if order:
            exchange.process_order(order, verbose=True)
            print(f"Time: {current_time}, Trader: {trader.tid}, Posted Order: {order}")

    # Print the tape (transaction log) when a trade occurs
    for trade in exchange.tape:
        print(f"Trade executed at Time: {trade['time']} for Price: {trade['price']} between {trade['party1']} and {trade['party2']}")

    # Wait for a short interval to mimic real-time delay (e.g., 1 second)
    time.sleep(1)
    current_time += 1
