import matplotlib.pyplot as plt
import numpy as np
import csv
import random

from BSE import market_session


start_time = 0
end_time = 60 * 5
chart1_range=(10, 200)

supply_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]

order_interval = 1
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'periodic'}

sellers_spec = [('ZIP', 3)]
buyers_spec = [('ZIP', 50)]
traders_spec = {'sellers':sellers_spec,
                'buyers':buyers_spec}

verbose = True

trial_id = 'Test'
dump_flags = {'dump_blotters': True,
              'dump_lobs': True,
              'dump_strats': True,
              'dump_avgbals': True,
              'dump_tape': True}

market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

prices_fname = trial_id + '_tape.csv'
x = np.empty(0)
y = np.empty(0)
with open(prices_fname, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        time = float(row[1])
        price = float(row[2])
        x = np.append(x,time)
        y = np.append(y,price)

plt.plot(x, y, '.', color='black')
plt.xlabel('Time/s')
plt.ylabel('Price/£')
plt.show()