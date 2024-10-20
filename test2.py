import matplotlib.pyplot as plt
import numpy as np
import csv
import random


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

order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'periodic'}



