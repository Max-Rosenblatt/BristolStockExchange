import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session
import os
import multiprocessing
import math



plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})







# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


def schedule_offsetfn(t):
    global market_state

    # Update volatility relative to current price (scale with price)
    market_state['volatility'] = 5e-6 * market_state['price']

    # Standard price update: Brownian motion with drift
    var = np.random.normal(loc=market_state['drift'], scale=market_state['volatility'])
    market_state['price'] += var

    # Trigger a major crash at a fixed time (t = 300 seconds)
    if (not market_state.get('crash_active', False) and
            market_state.get('crash_start_time') is None and
            math.floor(t) == 150):
        market_state['crash_active'] = True
        market_state['crash_start_time'] = t
        market_state['crash_price'] = market_state['price']
        # Use a random factor to simulate variability in crash severity (e.g., 20-40% drop)
        crash_jump_factor = 0.8
        market_state['price'] = market_state['crash_price'] * crash_jump_factor
        market_state['negative_drift_duration'] = 60  # seconds of extra downward pressure (panic selling)

        # Exponential recovery toward the pre-crash price
        recovery_strength = 5e-6
        target_price = market_state['crash_price']
        market_state['price'] += (target_price - market_state['price']) * recovery_strength

        # When price nears the pre-crash level, end the crash period
        if market_state['price'] >= target_price * 0.98:
            market_state['crash_active'] = False

    return int(round(market_state['price'], 0))




def run_simulation(args):
    # Unpack arguments:
    sim_num, buyer_type, seller_type, n_traders = args
    # Reset global market_state at the start of each simulation
    global market_state
    market_state = {
        'price': 100,  # Initial price level
        'drift': 9e-4,  # Small upward drift
        'crash_active': False,
        'crash_start_time': None,
    }

    # Simulation parameters
    rangeS = (100, 150, schedule_offsetfn)
    rangeD = rangeS

    # Simulation parameters
    start_time = 0
    end_time = 300
    order_interval = 1
    window = 45
    time_window_length = f'{window}s'
    shock_time = 150

    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeS], 'stepmode': 'jittered'}]
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeD], 'stepmode': 'jittered'}]

    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': order_interval,
        'timemode': 'drip-fixed'
    }

    # Create a unique trial id that encodes buyer and seller types
    trial_id = f"{buyer_type}_{seller_type}_{n_traders}trad_{sim_num}"

    # Trader specifications: use separate types for buyers and sellers
    traders_spec = {
        'buyers': [(buyer_type, n_traders)],
        'sellers': [(seller_type, n_traders)]
    }

    # Run market session:
    market_session(trial_id, start_time, end_time, traders_spec, order_sched,
                   {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False,
                    'dump_avgbals': False, 'dump_tape': True}, False)

    # Process the trades output:
    trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
    trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
    trades_df = trades_df.sort_values('Time').set_index('Time')

    start_time = trades_df.index.min()
    trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()

    trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1))
    trades_df['returns'] = trades_df['returns'].dropna()

    n_bins = 20000
    bins = np.linspace(trades_df['returns'].min() * 1.5, trades_df['returns'].max() * 1.5, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    trades_df['Entropy'] = np.nan
    valid_idx = trades_df[trades_df['Elapsed'] >= pd.to_timedelta(time_window_length).total_seconds()].index

    # Perform rolling on the full series first, then extract valid values
    rolling_entropy = trades_df['returns_binned'].rolling(
        time_window_length, closed='both'
    ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

    # Assign the computed entropy values to the DataFrame
    trades_df['Entropy'] = rolling_entropy

    trades_df.loc[trades_df['Elapsed'] < pd.to_timedelta(time_window_length).total_seconds(), 'Entropy'] = np.nan

    # Add a simple column for elapsed time in seconds:
    trades_df['Time_seconds'] = trades_df.index.total_seconds()

    # Define pre- and post-shock periods
    pre_shock = trades_df[(trades_df['Time_seconds'] < shock_time)]
    post_shock = trades_df[(trades_df['Time_seconds'] >= shock_time) & (trades_df['Time_seconds'] < (shock_time+30))]

    # Compute mean and standard error for pre-shock entropy:
    pre_entropy_vals = pre_shock['Entropy']
    pre_avg = pre_entropy_vals.mean()
    pre_stderr = pre_entropy_vals.std() / np.sqrt(len(trades_df)) if len(trades_df) > 0 else 0

    # Compute mean and standard error for post-shock entropy:
    post_entropy_vals = post_shock['Entropy']
    post_avg = post_entropy_vals.mean()
    post_stderr = post_entropy_vals.std() / np.sqrt(len(trades_df)) if len(trades_df) > 0 else 0

    # Determine the maximum entropy value over the simulation and its corresponding time:
    if trades_df['Entropy'].dropna().empty:
        max_entropy = np.nan
        max_time = np.nan
    else:
        max_entropy = trades_df['Entropy'].max()
        max_time = trades_df.loc[trades_df['Entropy'].idxmax(), 'Time_seconds']


    # (Optional) Generate a dual-axis plot for the current trial:
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.scatter(trades_df['Time_seconds'], trades_df['Price'], marker='x', color='black', alpha=0.7, label='Trades')
    ax1.set_xlabel("Time/s")
    ax1.set_ylabel("Trade Price")
    ax1.tick_params(axis='y')
    ax1.set_ylim(150, 300)

    ax2 = ax1.twinx()
    ax2.plot(trades_df['Time_seconds'], trades_df['Entropy'], ls='--', color='red', label='Entropy', lw=1)
    ax2.set_ylabel("Entropy/bits")
    ax2.tick_params(axis='y')
    ax2.set_ylim(0,10)
    plt.tight_layout()
    plt.show()



    # Clean up intermediate file
    os.remove(f'{trial_id}_tape.csv')

    # Return a tuple with all relevant results:
    return (sim_num, buyer_type, seller_type, pre_avg, pre_stderr, post_avg, post_stderr, max_entropy, max_time)


def run_parallel_simulations(n_simulations, n_traders, trader_types, output_filename):
    # Create output file with header
    with open(output_filename, 'w') as f:
        header = ("simulation_number,buyer_type,seller_type,"
                  "average_entropy_pre_shock,entropy_pre_shock_stderr,"
                  "average_entropy_post_shock,entropy_post_shock_stderr,"
                  "maximum_entropy,maximum_entropy_time\n")
        f.write(header)

    # Create list of all simulation parameters (16 combinations × n_simulations each)
    args_list = []
    for buyer_type in trader_types:
        for seller_type in trader_types:
            for sim in range(1, n_simulations + 1):
                args_list.append((sim, buyer_type, seller_type, n_traders))

    num_processes = multiprocessing.cpu_count()
    print(f"Running {len(args_list)} simulations using {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for count, result in enumerate(pool.imap_unordered(run_simulation, args_list), start=1):
            (sim_num, b_type, s_type, pre_avg, pre_stderr, post_avg, post_stderr, max_entropy, max_time) = result
            with open(output_filename, 'a') as f:
                f.write(
                    f"{sim_num},{b_type},{s_type},{pre_avg:.4f},{pre_stderr:.4f},{post_avg:.4f},{post_stderr:.4f},{max_entropy:.4f},{max_time:.2f}\n")
            print(
                f"Completed {count}/{len(args_list)}: Buyer={b_type}, Seller={s_type}, Sim={sim_num} | Pre: {pre_avg:.4f} ± {pre_stderr:.4f}, Post: {post_avg:.4f} ± {post_stderr:.4f}")

    print(f"\nAll results saved to {output_filename}")


# Main configuration
if __name__ == '__main__':
    n_simulations = 1  # 100 simulations per buyer-seller combination
    n_traders = 50

    # Define the four trader types for buyers and sellers
    trader_types = ['ZIP']



    output_filename = f'advancedshock_entropy_results_{n_simulations}sims_{n_traders}trad_allCombinations.csv'

    if os.path.exists(output_filename):
        response = input(f"File {output_filename} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting simulation.")
        else:
            run_parallel_simulations(n_simulations, n_traders, trader_types, output_filename)
    else:
        run_parallel_simulations(n_simulations, n_traders, trader_types, output_filename)

    # (Optional) You can load and plot aggregated results from the CSV file after simulations
    sim_df = pd.read_csv(output_filename)

    # Plot individual pre and post shock entropy values across simulations (for a rough overview)
    plt.figure(figsize=(10, 5))
    plt.plot(sim_df['average_entropy_pre_shock'], 'bx', label='Pre-Shock Entropy')
    plt.plot(sim_df['average_entropy_post_shock'], 'rx', label='Post-Shock Entropy')
    plt.xlabel('Simulation Index')
    plt.ylabel('Entropy (bits)')
    plt.legend()
    plt.show()

    # Plot convergence of the average entropies over the simulations:
    plt.figure(figsize=(10, 5))
    cumulative_pre = sim_df['average_entropy_pre_shock'].expanding().mean()
    cumulative_post = sim_df['average_entropy_post_shock'].expanding().mean()
    plt.plot(cumulative_pre, 'blue', label='Cumulative Pre-Shock Avg')
    plt.plot(cumulative_post, 'red', label='Cumulative Post-Shock Avg')
    plt.xlabel('Simulation Index')
    plt.ylabel('Average Entropy (bits)')
    plt.title('Convergence of Average Entropy Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
