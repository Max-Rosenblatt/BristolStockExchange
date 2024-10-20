import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import time

from BSE import market_session

start_time = 0
end_time = 60 * 10

chart1_range=(80, 320)

supply_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time,
                    'to': end_time,
                    'ranges': [chart1_range],
                    'stepmode': 'fixed'}]

order_interval = 10
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'periodic'}

sellers_spec = [('ZIP', 11)]
buyers_spec = sellers_spec
traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

verbose = False

trial_id = 'test'
dump_flags = {'dump_blotters': True, 'dump_lobs': True, 'dump_strats': True,
              'dump_avgbals': True, 'dump_tape': True}

market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)