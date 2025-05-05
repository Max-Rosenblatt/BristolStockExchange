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
    sim_num, n_traders, buyer_type, seller_type, output_file, price_range = args
    range1, range2 = price_range, price_range

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

    trial_id = f'{n_traders}_{seller_type}S_{buyer_type}B_{sim_num}'
    traders_spec = {
        'sellers': [(seller_type, n_traders)],
        'buyers': [(buyer_type, n_traders)]
    }

    market_session(
        trial_id,
        start_time,
        end_time,
        traders_spec,
        order_sched,
        {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True},
        False
    )

    try:
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

        rolling_entropy = trades_df['returns_binned'].rolling(
            time_window_length, closed='both'
        ).apply(lambda x: calculate_price_entropy(x.dropna()), raw=False)

        # Assign the computed entropy values to the DataFrame
        trades_df['Entropy'] = rolling_entropy

        trades_df.loc[trades_df['Elapsed'] < pd.to_timedelta(time_window_length).total_seconds(), 'Entropy'] = np.nan

        # Add a simple column for elapsed time in seconds:
        trades_df['Time_seconds'] = trades_df.index.total_seconds()

        avg_entropy = trades_df['Entropy'].mean()
        entropy_std = trades_df['Entropy'].std()
        entropy_stderr = entropy_std / np.sqrt(len(trades_df)) if len(trades_df) > 0 else 0
        num_trades = len(trades_df)

    except Exception as e:
        print(f"Error processing {trial_id}: {str(e)}")
        avg_entropy = 0
        entropy_stderr = 0
        num_trades = 0
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.scatter(trades_df['Time_seconds'], trades_df['Price'], marker='x', color='black', alpha=0.7, label='Trades')
    ax1.set_xlabel("Time/s")
    ax1.set_ylabel("Trade Price")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.plot(trades_df['Time_seconds'], trades_df['Entropy'], ls='--', color='red', label='Entropy', lw=1)
    ax2.set_ylabel("Entropy/bits")
    ax2.tick_params(axis='y')
    plt.tight_layout()
    plt.show()
    """

    if os.path.exists(f'{trial_id}_tape.csv'):
        os.remove(f'{trial_id}_tape.csv')

    print(f'Completed {trial_id}')
    return sim_num, buyer_type, seller_type, avg_entropy, entropy_stderr, num_trades


def plot_results(df, trader_types):
    # Plot cumulative mean entropy with confidence intervals for all combinations
    plt.figure(figsize=(14, 10))
    combinations = df[['buyer_type', 'seller_type']].drop_duplicates().values

    for buyer_type, seller_type in combinations:
        subset = df[(df['buyer_type'] == buyer_type) & (df['seller_type'] == seller_type)].sort_values(
            'simulation_number')

        if len(subset) > 0:
            entropies = subset['average_entropy'].values
            sim_numbers = subset['simulation_number'].values

            cumulative_mean = np.cumsum(entropies) / np.arange(1, len(entropies) + 1)
            cumulative_std = [np.std(entropies[:i + 1], ddof=1) if i > 0 else 0 for i in range(len(entropies))]
            cumulative_ci = [1.96 * (std / np.sqrt(i + 1)) if i > 0 else 0 for i, std in enumerate(cumulative_std)]

            plt.plot(sim_numbers, cumulative_mean, label=f'{buyer_type} buyers, {seller_type} sellers')
            upper_bound = cumulative_mean + cumulative_ci
            lower_bound = cumulative_mean - cumulative_ci
            plt.fill_between(sim_numbers, lower_bound, upper_bound, alpha=0.2)

    plt.xlabel("Number of Simulations")
    plt.ylabel("Cumulative Average Entropy (bits)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('entropy_all_combinations.png', bbox_inches='tight')
    plt.show()

    # Calculate required simulations for precision
    print("\nRequired simulations for precision:")
    for buyer_type, seller_type in combinations:
        subset = df[(df['buyer_type'] == buyer_type) & (df['seller_type'] == seller_type)]
        entropies = subset['average_entropy'].values
        if len(entropies) > 1:
            sample_variance = np.var(entropies, ddof=1)
            z_score = norm.ppf(0.975)
            epsilon = 0.05
            required_simulations = int(np.ceil((sample_variance * z_score ** 2) / (epsilon ** 2)))
            print(f"{buyer_type} buyers, {seller_type} sellers: {required_simulations}")


def main():
    trader_types = ['ZIC', 'ZIP', 'GVWY', 'SHVR']
    n_simulations = 100
    n_traders = 50
    price_range = (100, 150)
    output_filename = 'entropy_by_trader_types_50_v2.csv'

    if os.path.exists(output_filename):
        user_response = input(f"The file '{output_filename}' exists. Overwrite? (y/n): ").strip().lower()
        if user_response != 'y':
            print("Operation aborted.")
    if os.path.exists(output_filename):
        existing_df = pd.read_csv(output_filename)
    else:
        existing_df = pd.DataFrame(columns=[
            'simulation_number', 'buyer_type', 'seller_type',
            'average_entropy', 'entropy_stderr', 'num_trades'
        ])

    all_combinations = {(sim, bt, st)
                        for bt in trader_types
                        for st in trader_types
                        for sim in range(1, n_simulations + 1)}

    existing_combinations = set(zip(
        existing_df['simulation_number'],
        existing_df['buyer_type'],
        existing_df['seller_type']
    ))

    missing_combinations = list(all_combinations - existing_combinations)

    if missing_combinations:
        print(f"Running {len(missing_combinations)} missing simulations...")
        args_list = [
            (sim, n_traders, bt, st, output_filename, price_range)
            for sim, bt, st in missing_combinations
        ]

        with Pool(processes=(cpu_count()*2)) as pool:
            results = pool.map(run_simulation, args_list)

        new_df = pd.DataFrame(results, columns=[
            'simulation_number', 'buyer_type', 'seller_type',
            'average_entropy', 'entropy_stderr', 'num_trades'
        ])

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.sort_values(by=['buyer_type', 'seller_type', 'simulation_number'], inplace=True)
        combined_df.to_csv(output_filename, index=False)
    else:
        print(f"All {len(all_combinations)} simulations already complete.")

    df = pd.read_csv(output_filename)
    plot_results(df, trader_types)

    # Generate summary statistics
    summary_df = df.groupby(['buyer_type', 'seller_type']).agg({
        'average_entropy': ['mean', 'std', 'count'],
        'num_trades': ['mean', 'std'],
        'entropy_stderr': ['mean']
    }).reset_index()

    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    summary_df['average_entropy_stderr'] = summary_df['average_entropy_std'] / np.sqrt(
        summary_df['average_entropy_count'])
    summary_df['num_trades_stderr'] = summary_df['num_trades_std'] / np.sqrt(summary_df['average_entropy_count'])

    print("\nSummary Statistics:")
    print(summary_df.to_string())
    summary_df.to_csv('entropy_summary_statistics.csv', index=False)


if __name__ == '__main__':
    main()