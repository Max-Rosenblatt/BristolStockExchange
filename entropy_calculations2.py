import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from BSE import market_session


# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


# Simulation parameters
start_time = 0
end_time = 60
chart1_range = (1, 100)
order_interval = 1

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

# Fixed parameters
n = 50
seller_types = ['ZIC', 'ZIP']
num_simulations = 10  # Number of times to run each setup
bin_sizes = np.linspace(10, 5000, 100, dtype=int)
window_length = 10  # Fixed rolling window


# Function to run a single simulation
def run_simulation(params):
    bin_size, seller_type, sim = params
    buyer_type = 'ZIP'
    trial_id = f'test_{seller_type}_{sim}_{bin_size}'

    sellers_spec = [(seller_type, n)]
    buyers_spec = [(buyer_type, n)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

    dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False, 'dump_avgbals': False,
                  'dump_tape': True}
    verbose = False

    # Run market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

    # Read in market data
    trades_file = f'{trial_id}_tape.csv'
    trades_df = pd.read_csv(trades_file)
    trades_df.columns = ['Type', 'Time', 'Price']
    trades_df['id'] = range(len(trades_df))
    trades_df.set_index('id', inplace=True)
    trades_df = trades_df.sort_values(by="Time")
    trades_df['returns'] = trades_df['Price'].pct_change().dropna()

    # Bin returns and calculate entropy
    bins = np.linspace(-10, 10, bin_size)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
    trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=window_length).apply(
        lambda x: calculate_price_entropy(x.dropna()))

    avg_entropy = trades_df['Entropy'].mean()
    print(f"Traders: {n}, Seller: {seller_type}, Bin Size: {bin_size}, Simulation: {sim}, Avg Entropy: {avg_entropy}")

    return (n, seller_type, sim, bin_size, avg_entropy)


# Multiprocessing setup
if __name__ == '__main__':
    num_workers = mp.cpu_count() - 1  # Use all but one CPU core
    pool = mp.Pool(num_workers)

    # Generate parameter combinations
    params_list = [(bin_size, seller_type, sim) for bin_size in bin_sizes for seller_type in seller_types for sim in
                   range(num_simulations)]

    # Run simulations in parallel
    results = pool.map(run_simulation, params_list)

    # Close the pool
    pool.close()
    pool.join()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Num Traders', 'Seller Type', 'Simulation', 'Bin Size', 'Avg Entropy'])

    # Plot results
    plt.figure(figsize=(10, 5))
    for seller in seller_types:
        subset = results_df[results_df['Seller Type'] == seller]
        plt.plot(subset['Bin Size'], subset['Avg Entropy'], marker='o', linestyle='', label=seller)
    plt.xlabel("Bin Size")
    plt.ylabel("Average Entropy")
    plt.title("Market Entropy Across Multiple Simulations")
    plt.legend()
    plt.show()
