import numpy as np
import csv
import random
from BSE import market_session

import matplotlib.pyplot as plt
import numpy as np
import csv
import random

import BSE


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

order_interval = 10
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'drip-poisson'}

sellers_spec = [('ZIP', 30)]
buyers_spec = [('ZIP', 50)]
traders_spec = {'sellers':sellers_spec,
                'buyers':buyers_spec}

verbose = False

trial_id = 'Test'
dump_flags = {'dump_blotters': True,
              'dump_lobs': True,
              'dump_strats': True,
              'dump_avgbals': True,
              'dump_tape': True}

public_data = BSE.market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

print(public_data)