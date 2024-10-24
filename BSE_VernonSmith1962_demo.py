#%% md
# # Simple BSE demo/walkthrough
# Dave Cliff, University of Bristol, October 2022
# 
# 
# 
#%% md
# ## BSE System Architecture
# 
# The figure below shows a schematic illustration of the overall architecture of the *Bristol Stock Exchange* (BSE), a simulation of a contemporary fully electronic financial exchange with automated traders.
#%% md
# ![BSE_system_diagram.png](attachment:8626ec3c-5c9c-4ae1-821a-31ee98123508.png)
#%% md
# BSE simulates a *market*, composed of an *exchange* and some number *N* of *traders* which each interact with the exchange. Any one simulation of a market session proceeds according to BSE's *system clock*, which provides a unified time signal to all elements of the simulation.
# 
# Separate from the simulation of the market is BSE's *session control* logic, which determines the (potentially time-dependent) market's *supply and demand schedule* (SDS): this is used to issue *assignments* to the traders, i.e. allocations of cash and limit-prices to buyers, and allocations of stock and limit-prices to sellers -- this is the simulation's correlate of real-world market *customer orders* coming from customers to sales-traders who are responsible for working each customer order. The session-control logic is also responsible for recording whole-market data, such as the profits and strategy-values of each trader in the market, as were visualised in the graphs and plots earlier in this paper.
# 
# The exchange receives orders from the traders: *bids* from buyers; *asks* from sellers. When each order arrives at the exchange, it is processed by the *matching engine*, attempting to find one or more matching bids for a newly-arrived ask, or one or more matching asks for a newly-arrived bid. It does this by comparing the new order to those earlier orders, as yet unfulfilled, that are "resting" at the exchange and which are summarised in aggregated and anonymized form on the exchange's *limit order book* (LOB). If a new order can be matched with one or more existing orders on the LOB then the matching orders are removed from the LOB, and the new order plus its counterparty orders from the LOB are recorded as fulfilled, resulting in a transaction taking place. When a transaction occurs, its details are written to the exchange's public record of transactions which is commonly referred to as the exchange's *tape* -- the tape records transactions and also other notable market events, such as cancellations of existing orders. When a transaction occurs, the exchange also notifies the traders concerned, adjusting their cash balances appropriately. The BSE exchange also can be configured to write the state of the LOB at any one instance (referred to as a *LOB frame*) to an external record, the *LOB framestore*, for subsequent analysis.
# 
# Each of the *N* traders in the market receives occasional fresh assignments from the session control, all have the same view of the LOB data published by the exchange, and when a trader is involved in a transaction it receives notification of the relevant details from the exchange's matching engine. Each trader is able to send orders to the exchange, and to send cancellations of existing orders, and each maintains its own local private record of assignments received, orders sent to the exchange, and transaction details received from the exchange: this is conventionally referred to as the trader's *blotter*.
# 
#%% md
# ## Using BSE: a first walk-through
# 
# Let's start by using BSE to replicate the experiment that Vernon Smith showed results from in Chart 1 of his landmark 1962 JPE paper *"An Experimental Study of Competitive Market Behavior"* -- this was Smith's first reported CDA experiment.
# 
# For a PDF version of Smith's 1962 paper, see here: https://www.journals.uchicago.edu/doi/abs/10.1086/258609
# 
# First of all let's import what we'll need.
#%%
# un-comment these lines if you need to install the packages
# !{sys.executable} pip3 install numpy
# !{sys.executable} pip3 install matplotlib

import matplotlib.pyplot as plt
import numpy as np
import csv
import random

from BSE import market_session
#%% md
# Let's say all our experiments are to last for 10 simulated minutes...
#%%
start_time = 0
end_time = 60 * 10
#%% md
# The supply and demand curves in Smith's Chart 1 are symmetric: they each involve 11 traders, each given an assignment to trade (buy or sell) a single unit.
# 
# On both curves the minimum price was 80 and the max was 320,
# and the step-size between successive prices was always 20
#%%
chart1_range=(80, 320)

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
#%% md
# Smith used periodic updating -- at the start of each "day" all traders are issued with fresh assignments.
# 
# Let's do that once every 60 seconds
#%%
order_interval = 60
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': order_interval, 'timemode': 'periodic'}
#%% md
# And finally let's use 11 ZIP traders on each side, buyers and sellers, because ZIP gives reasonably human-like market dynamics.
#%%
sellers_spec = [('ZIP', 11)]
buyers_spec = sellers_spec
traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
#%% md
# When we run a BSE market session we can alter how verbose it is, how much it tells us about what is going on, but this can generate a *lot* of text, so let's switch that off for the time being.
#%%
verbose = False
#%% md
# Finally BSE (which was originally created before the days of Jupyter notebooks) writes output data-files
# in csv format for later anaylsis, so we need to give it a session-identifier string which will be used at the start of all this session's data-files, and what data to dump into files.
# 
#%%

trial_id = 'Test'
dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
              'dump_avgbals': False, 'dump_tape': True}
#%% md
# And now we're ready to go... we'll run a market session, which dumps data to a file, and then we'll immediately read the file back and plot a graph of the transaction-price time-series.
# 
#%%

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

plt.plot(x, y, 'x', color='black');
plt.show()