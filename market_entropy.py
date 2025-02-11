import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))



# Main function
if __name__ == "__main__":

    # Simulation parameters
    start_time = 0
    end_time = 60
    chart1_range = (1, 100)
    order_interval = 1

    supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
    demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
    order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval,
                   'timemode': 'periodic'}

    # Loop over different numbers of traders
    num_traders = np.linspace(50, 50, 1, dtype=int)
    bins = np.linspace(100, 1000, 10, dtype=int)
    for n in num_traders:

        # Run the market session (assuming market_session is already defined in your environment)
        from BSE import market_session


        # Setup buyer and seller types
        buyer_type = 'ZIP'
        seller_type = 'ZIC'

        trial_id = f'{n}_traders_{buyer_type}_{seller_type}'

        sellers_spec = [(seller_type, n)]
        buyers_spec = [(buyer_type, n)]

        traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

        # Define additional arguments for market_session
        dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True,
                      'dump_tape': True}
        verbose = False

        # Run the market session with the given parameters
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

        # Wait for the LOB frames CSV to be created and populated
        l2_data = trial_id + '_LOB_frames.csv'

        # Read in the market data
        market_data = pd.read_csv(l2_data)

        # Data manipulation for entropy calculation
        trades_file = f'{trial_id}_tape.csv'

        trades_df = pd.read_csv(trades_file)
        trades_df.columns = ['Type', 'Time', 'Price']
        trades_df['id'] = range(len(trades_df))
        trades_df.set_index('id', inplace=True)
        trades_df = trades_df.sort_values(by="Time")
        trades_df['returns'] = trades_df['Price'].pct_change().dropna()

        # Bin returns and calculate entropy
        bins = np.linspace(-10, 10, 1000)
        trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
        trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=15).apply(
            lambda x: calculate_price_entropy(x.dropna()))

        # Plotting Entropy and Price vs Time
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axs[1].plot(trades_df['Time'], trades_df['Entropy'], ls='', marker='x', color='navy')
        axs[0].plot(trades_df['Time'], trades_df['Price'], ls='', marker='x', color='black')

        axs[1].set_title(f"Entropy vs Time - {n} Traders")
        axs[0].set_title(f"Price vs Time - {n} Traders")

        axs[1].set_xlabel(f"Time")

        axs[1].set_ylabel(f"Shannon Entropy/Bits")
        axs[0].set_ylabel(f"Price")

        plt.tight_layout()
        plt.show()
