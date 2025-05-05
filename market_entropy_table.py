import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session
import os
from multiprocessing import Pool, cpu_count
from scipy.stats import norm

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})


def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq)) if len(data) > 0 else 0


def run_simulation(args):
    sim_num, n_traders, buyer_type, output_file, range1, range2 = args

    start_time = 0
    end_time = 300
    order_interval = 1
    window = 45
    time_window_length = f'{window}s'

    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'jittered'}]
    demand_schedule = supply_schedule
    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': order_interval,
        'timemode': 'drip-fixed'
    }

    trial_id = f'{n_traders}_{buyer_type}B_GVWYS_{sim_num}'
    traders_spec = {
        'sellers': [('GVWY', n_traders)],
        'buyers': [(buyer_type, n_traders)]
    }

    # Run the market session
    market_session(
        trial_id,
        start_time,
        end_time,
        traders_spec,
        order_sched,
        {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True},
        False
    )

    # Read and process trades
    trades_df = pd.read_csv(f'{trial_id}_tape.csv', names=['Type', 'Time', 'Price'])
    trades_df['Time'] = pd.to_timedelta(trades_df['Time'], unit='s')
    trades_df = trades_df.sort_values('Time').set_index('Time')

    start_time = trades_df.index.min()
    trades_df['Elapsed'] = (trades_df.index - start_time).total_seconds()

    trades_df['returns'] = np.log(trades_df['Price'] / trades_df['Price'].shift(1)).dropna()
    n_bins = 2000
    bins = np.linspace(trades_df['returns'].min() * 1.5, trades_df['returns'].max() * 1.5, n_bins)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

    trades_df['Entropy'] = np.nan
    valid_idx = trades_df[trades_df['Elapsed'] >= pd.to_timedelta(time_window_length).total_seconds()].index

    rolling_entropy = trades_df.loc[valid_idx, 'returns_binned'].rolling(
        time_window_length, closed='both'
    ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

    trades_df.loc[rolling_entropy.index, 'Entropy'] = rolling_entropy
    trades_df['Time_seconds'] = trades_df.index.total_seconds()

    avg_entropy = trades_df['Entropy'].mean()

    os.remove(f'{trial_id}_tape.csv')
    print(f'completed {trial_id}')

    num_trades = len(trades_df)
    return sim_num, buyer_type, avg_entropy, num_trades



def plot_results(df, buyer_types):
    # Plot entropy vs simulation number
    plt.figure(figsize=(12, 6))
    for buyer_type in buyer_types:
        subset = df[df['buyer_type'] == buyer_type]
        plt.plot(subset['simulation_number'], subset['average_entropy'], marker='x', ls='-', label=buyer_type)

    plt.xlabel("Simulation Number")
    plt.ylabel("Average Entropy (bits)")
    plt.title("Entropy vs Simulation Number by Buyer Type")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot cumulative mean entropy
    plt.figure(figsize=(12, 6))
    for buyer_type in buyer_types:
        subset = df[df['buyer_type'] == buyer_type].sort_values('simulation_number')
        subset['Cumulative_Mean_Entropy'] = subset['average_entropy'].expanding().mean()
        plt.plot(
            subset['simulation_number'],
            subset['Cumulative_Mean_Entropy'],
            label=buyer_type
        )

    plt.xlabel("Simulation Number")
    plt.ylabel("Cumulative Average Entropy (bits)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot with confidence intervals
    plt.figure(figsize=(14, 7))
    for buyer_type in buyer_types:
        subset = df[df['buyer_type'] == buyer_type].sort_values('simulation_number')
        entropies = subset['average_entropy'].values
        sim_numbers = subset['simulation_number'].values

        cumulative_mean = np.cumsum(entropies) / np.arange(1, len(entropies) + 1)
        cumulative_std = [np.std(entropies[:i + 1], ddof=1) if i > 0 else 0 for i in range(len(entropies))]
        cumulative_ci = [1.96 * (std / np.sqrt(i + 1)) if i > 0 else 0 for i, std in enumerate(cumulative_std)]

        plt.plot(sim_numbers, cumulative_mean, label=f'{buyer_type}')
        upper_bound = cumulative_mean + cumulative_ci
        lower_bound = cumulative_mean - cumulative_ci
        plt.fill_between(sim_numbers, lower_bound, upper_bound, alpha=0.2)

    plt.xlabel("Number of Simulations")
    plt.ylabel("Cumulative Average Entropy/bits")
    plt.legend()
    plt.tight_layout()
    plt.savefig('entropy_noshock_alltraders.png')
    plt.show()

    # Calculate required simulations for precision
    for buyer_type in buyer_types:
        entropies = df[df['buyer_type'] == buyer_type]['average_entropy'].values
        if len(entropies) > 1:
            sample_variance = np.var(entropies, ddof=1)
            z_score = norm.ppf(0.975)
            epsilon = 0.05
            required_simulations = int(np.ceil((sample_variance * z_score ** 2) / (epsilon ** 2)))
            print(
                f"{buyer_type}: Estimated simulations needed for ±{epsilon} precision at 95% confidence: {required_simulations}")


if __name__ == '__main__':
    buyer_types = ['ZIC', 'ZIP', 'GVWY', 'SHVR']
    n_simulations = 100
    n_traders = 50
    price_range = (100, 150)
    output_filename = 'entropy_by_buyer_type_50_jittered.csv'

    # Handle existing results file
    if os.path.exists(output_filename):
        user_response = input(f"The file '{output_filename}' exists. Overwrite? (y/n): ").strip().lower()
        if user_response != 'y':
            print("Operation aborted.")

    # Load existing results or create new dataframe
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
    else:
        existing_df = pd.DataFrame(columns=['simulation_number', 'buyer_type', 'average_entropy', 'num_trades'])

    # Determine missing simulations
    all_combinations = {(sim, bt) for bt in buyer_types for sim in range(1, n_simulations + 1)}
    existing_combinations = set(zip(existing_df['simulation_number'], existing_df['buyer_type']))
    missing_combinations = list(all_combinations - existing_combinations)

    if missing_combinations:
        print(f"Running {len(missing_combinations)} missing simulations...")
        args_list = [
            (sim, n_traders, bt, output_filename, price_range, price_range)
            for sim, bt in missing_combinations
        ]

        with Pool(processes=min(cpu_count(), len(args_list))) as pool:
            results = pool.map(run_simulation, args_list)

        new_df = pd.DataFrame(results, columns=['simulation_number', 'buyer_type', 'average_entropy', 'num_trades'])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.sort_values(by=['buyer_type', 'simulation_number'], inplace=True)
        combined_df.to_csv(output_filename, index=False)
    else:
        print(f"All {len(all_combinations)} simulations already complete.")

    # Load complete data and plot
    df = pd.read_csv(output_filename)
    plot_results(df, buyer_types)


# Load the data
df = pd.read_csv("entropy_by_buyer_type_50_jittered.csv")


# Group by buyer_type and compute average number of trades
trade_summary = df.groupby("buyer_type")['num_trades'].agg(['mean', 'std', 'count']).reset_index()
trade_summary['stderr'] = trade_summary['std'] / np.sqrt(trade_summary['count'])

print("\nAverage number of trades by buyer type (with standard error):")
for _, row in trade_summary.iterrows():
    print(f"{row['buyer_type']: <10} {row['mean']:.1f} ± {row['stderr']:.1f}")

total_trades_by_type = df.groupby("buyer_type")["num_trades"].sum()
print("\nTotal number of trades by buyer type:")
print(total_trades_by_type)
