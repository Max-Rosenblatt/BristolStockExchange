import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session
import os
import multiprocessing

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})




def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0


def run_simulation(args):
    sim_num, n_traders, trader_type, output_file, range1, range2 = args

    # Simulation parameters
    start_time = 0
    end_time = 300
    range1 = range1
    range2 = range2
    order_interval = 1
    window = 45
    time_window_length = f'{window}s'

    shock_time = 150
    pre_start = shock_time - window
    pre_end = shock_time
    post_start = shock_time
    post_end = shock_time + window

    # Configure market session
    supply_schedule = [
        {'from': 0, 'to': 150, 'ranges': [range1], 'stepmode': 'jittered'},
        {'from': 150, 'to': 300, 'ranges': [range2], 'stepmode': 'jittered'}
    ]
    demand_schedule = supply_schedule
    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': order_interval,
        'timemode': 'drip-fixed'
    }

    trial_id = f'{n_traders}_{trader_type}B_{trader_type}S_{sim_num}'
    traders_spec = {
        'sellers': [(trader_type, n_traders)],
        'buyers': [(trader_type, n_traders)]
    }

    # Run market session
    market_session(trial_id, start_time, end_time, traders_spec, order_sched,
                   {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False,
                    'dump_avgbals': False, 'dump_tape': True}, False)

    # Process results
    trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
    trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')  # Convert to timedelta
    trades_df = trades_df.sort_values('Time').set_index('Time')

    start_time = trades_df.index.min()
    trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()
    window_seconds = pd.to_timedelta(time_window_length).total_seconds()

    # Calculate returns and bin data
    trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1)).dropna()
    n_bins = 1000
    bins = np.linspace(trades_df['returns'].min() * 1.1, trades_df['returns'].max() * 1.1, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
    rolling_entropy = trades_df['returns_binned'].rolling(
        time_window_length, closed='both'
    ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

    trades_df['Entropy'] = np.nan  # Initialize with NaN

    # Filter to only include points where elapsed time >= window_seconds
    valid_mask = trades_df['Elapsed'] >= window_seconds
    filtered_rolling_entropy = rolling_entropy.where(valid_mask)

    # Store the result back - this will keep NaN for early times
    trades_df['Entropy'] = filtered_rolling_entropy

    # Convert index back to seconds for plotting
    trades_df['Time_seconds'] = trades_df.index.total_seconds()

    pre_shock = trades_df[(trades_df['Time_seconds'] >= pre_start) & (trades_df['Time_seconds'] < pre_end)]
    post_shock = trades_df[(trades_df['Time_seconds'] >= post_start) & (trades_df['Time_seconds'] < post_end)]

    pre_avg = pre_shock['Entropy'].mean() if not pre_shock['Entropy'].dropna().empty else np.nan
    post_avg = post_shock['Entropy'].mean() if not post_shock['Entropy'].dropna().empty else np.nan

    # Create visualization with dual axes
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Price on primary axis (left)
    ax1.scatter(trades_df['Time_seconds'], trades_df['Price'], marker='x', color='black', alpha=0.7, label='Trades')
    ax1.set_xlabel("Time/s")
    ax1.set_ylabel("Trade Price")
    ax1.tick_params(axis='y')

    # Create secondary axis (right) for Entropy
    ax2 = ax1.twinx()
    ax2.plot(trades_df['Time_seconds'], trades_df['Entropy'], ls='--', color='red', label='Entropy', lw=1)
    ax2.set_ylabel("Entropy/bits")
    ax2.tick_params(axis='y')
    plt.tight_layout()

    plt.savefig(f'{trial_id}.png')

    plt.show()

    # Clean up intermediate files
    os.remove(f'{trial_id}_tape.csv')

    return sim_num, pre_avg, post_avg


def run_parallel_simulations(n_simulations, n_traders, trader_type, output_filename, range1, range2):
    # Create output file with header
    with open(output_filename, 'w') as f:
        f.write("Simulation,Pre_Shock_Entropy,Post_Shock_Entropy\n")

    # Determine number of processes to use
    num_processes = multiprocessing.cpu_count()
    print(f"Running {n_simulations} simulations using {num_processes} processes...")

    # Create pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Prepare arguments for each simulation
        args = [(i + 1, n_traders, trader_type, output_filename, range1, range2) for i in range(n_simulations)]

        # Process results as they come in
        for i, (sim_num, pre, post) in enumerate(pool.imap_unordered(run_simulation, args)):
            # Write results to file immediately
            with open(output_filename, 'a') as f:
                f.write(f"{sim_num},{pre:.4f},{post:.4f}\n")
                print(f"Completed {i + 1}/{n_simulations} simulations. Pre = {pre:.4f}, Post = {post:.4f}")

    print(f"\nAll results saved to {output_filename}")


# Main configuration
n_simulations = 50  # Set the number of simulations to run
n_traders = 50



trader_type = 'ZIP'


if __name__ == '__main__':
    range1 = (100,150)
    range2 = (80,130)


    output_filename = f'entropy_results_{n_simulations}sims_{n_traders}trad_{trader_type}.csv'
    # Check if file exists and prompt user
    if os.path.exists(output_filename):
        #response = input(f"File {output_filename} already exists. Overwrite? (y/n): ")
        #if response.lower() != 'y':
        if 1==2:
            print("Aborting simulation.")
        else:
            run_parallel_simulations(n_simulations, n_traders, trader_type, output_filename, range1=range1, range2=range2)
    else:
        run_parallel_simulations(n_simulations, n_traders, trader_type, output_filename, range1=range1, range2=range2)

    # Load and plot results
    sim_df = pd.read_csv(output_filename)

    # Plot individual results
    plt.figure(figsize=(10, 5))
    plt.plot(sim_df['Pre_Shock_Entropy'], ls=' ', color='blue', marker='x', label='Pre-Shock Entropy')
    plt.plot(sim_df['Post_Shock_Entropy'], ls=' ', color='red', marker='x', label='Post-Shock Entropy')
    plt.xlabel('Simulation Number')
    plt.ylabel('Entropy/Bits')
    plt.legend()
    plt.show()

    # Plot convergence
    plt.figure(figsize=(10, 5))
    cumulative_pre = sim_df['Pre_Shock_Entropy'].expanding().mean()
    cumulative_post = sim_df['Post_Shock_Entropy'].expanding().mean()
    plt.plot(cumulative_pre, 'blue', label='Pre-Shock Avg')
    plt.plot(cumulative_post, 'red', label='Post-Shock Avg')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Average Entropy (bits)')
    plt.title('Convergence of Average Entropy Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()