import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from BSE import market_session

plt.rcParams['figure.dpi'] = 300
font = {'size'   : 18}

plt.rc('font', **font)

# Functions _________________________

def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Main Script _______________________
if __name__ == "__main__":
    output = pd.DataFrame(columns=['buyer', 'n_traders', 'avg_entropy'])

    # Set up simulation
    start_time = 0
    end_time = 60
    chart1_range = (1, 100)
    order_interval = 1

    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
    order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

    verbose = False
    dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True, 'dump_tape': True}
    num_traders = np.linspace(50, 50, 1, dtype=int)

    # Define buyer types
    buyer_types = ['ZIC', 'ZIP', 'GVWY', 'SHVR']  # Add all buyer types here

    # Create a single figure for combined histograms
    plt.figure(figsize=(12, 8))

    for buyer_type in buyer_types:
        for n in num_traders:
            seller_type = 'ZIC'
            trial_id = f'test_{buyer_type}'

            # Specify sellers and buyers based on the current number of traders
            sellers_spec = [(seller_type, n)]
            buyers_spec = [(buyer_type, n)]
            traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

            # Run the market session
            market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

            # Data manipulation
            trades_file = f'{trial_id}_tape.csv'
            trades_df = pd.read_csv(trades_file)
            trades_df.columns = ['Type', 'Time', 'Price']
            trades_df['id'] = range(len(trades_df))
            trades_df.set_index('id', inplace=True)
            trades_df = trades_df.sort_values(by="Time")
            trades_df['returns'] = trades_df['Price'].pct_change().dropna()

            bins = np.linspace(trades_df['returns'].min(), trades_df['returns'].max(), 1000)
            trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False, duplicates='drop')
            trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=25).apply(lambda x: calculate_price_entropy(x.dropna()))

            mean_entropy = trades_df['Entropy'].mean()
            new_row = {'buyer': buyer_type, 'n_traders': n * 2, 'avg_entropy': mean_entropy}
            output = pd.concat([output, pd.DataFrame([new_row])], ignore_index=True)
            output.to_csv('output.csv')

            # Plot histogram of binned returns
            plt.hist(
                trades_df['returns'] * 100,
                bins=50,
                alpha=0.5,
                label=f'{buyer_type}'
            )

    # Add labels, legend, and title
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.xlabel('Price Change between Adjacent Trades (%)')
    plt.ylabel('Frequency')
    plt.title('Price Change Distribution by Buyer Type')
    plt.legend(title="Buyer Types")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-100,200)

    # Show the combined plot
    plt.show()
