import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session

# Global state variables
market_state = {
    'price': 100,  # Initial price level
    'drift': 0.01,  # Small upward drift to reflect overall growth
    'volatility': 0.1,  # Volatility of price movement
    'crash_active': False,
    'crash_start_time': None,
    'crash_factor': 0.9
}

shock_times = []  # Store times when market crashes



# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

def schedule_offsetfn(t):
    global market_state

    # Brownian motion (random walk with drift)
    var = np.random.normal(loc=market_state['drift'], scale=market_state['volatility'])
    market_state['price'] += var  # Apply Brownian motion

    # Crash probability increases over time (e.g., exponential growth)
    crash_prob = 0.001

    # Trigger a crash
    if not market_state['crash_active'] and np.random.rand() < crash_prob:
        market_state['crash_active'] = True
        market_state['crash_start_time'] = t
        market_state['crash_price'] = market_state['price']  # Store price at crash
        market_state['price'] *= market_state['crash_factor']  # Instant drop
        shock_times.append(market_state['crash_start_time'])  # Track shock time
        print(f"Market crash at t={t}")

    # Apply gradual recovery using exponential approach
    if market_state['crash_active']:
        recovery_strength = 0.001

        # Apply recovery but also dampen randomness pulling it down too far
        target_price = market_state['crash_price']  # The price before crash
        market_state['price'] += (target_price - market_state['price']) * recovery_strength


        # Stop recovery when almost fully recovered
        if abs(market_state['price']/target_price) > 0.98: # Slightly relaxed condition
            market_state['crash_active'] = False

    return int(round(market_state['price'], 0))

rangeS = (100, 100, schedule_offsetfn)
rangeD = rangeS

# Simulation parameters
start_time = 0
end_time = 600
order_interval = 10

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeS], 'stepmode': 'jittered'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeD], 'stepmode': 'jittered'}]

order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-poisson'}

verbose = False
dump_flags = {
    'dump_blotters': False,
    'dump_lobs': True,
    'dump_strats': False,
    'dump_avgbals': False,
    'dump_tape': True
}

# Loop over different numbers of traders
num_traders = [100]
buyer_types = ['ZIP']
seller_types = buyer_types

for n in num_traders:
    for seller in seller_types:
        for buyer in buyer_types:
            trial_id = f'{n}_{buyer}B_{seller}S'
            buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
            sellers_spec = buyers_spec
            traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

            # Fixed parameters for bins and time window
            n_bins = 2000
            time_window = 30  # rolling window of 50 seconds

            # Run market session
            market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

            # Read in market data
            trades_file = f'{trial_id}_tape.csv'
            trades_df = pd.read_csv(trades_file)
            trades_df.columns = ['Type', 'Time', 'Price']
            trades_df['id'] = range(len(trades_df))
            trades_df['seller'] = seller
            trades_df.set_index('id', inplace=True)
            trades_df = trades_df.sort_values(by="Time")
            trades_df['returns'] = trades_df['Price'].pct_change().dropna()

            # Convert 'Time' column (in seconds) to a timedelta index for time-aware rolling
            trades_df['Time_dt'] = pd.to_timedelta(trades_df['Time'], unit='s')
            trades_df.set_index('Time_dt', inplace=True)

            # Bin returns for entropy calculation
            bins = np.linspace(trades_df['returns'].min() * 2, trades_df['returns'].max() * 2, n_bins)
            trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)

            # Calculate entropy using a rolling time window
            trades_df['Entropy'] = trades_df['returns_binned'].rolling(f'{time_window}s').apply(
                lambda x: calculate_price_entropy(x.dropna()), raw=False)

            # Save stats for further analysis (reset index to bring 'Time_dt' back as a column)
            trades_df_reset = trades_df.reset_index()
            trades_df_reset.to_csv(f'{trial_id}_stats.csv', index=False)


            # Create subplots for Price and Entropy
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Plot trade prices
            axes[0].scatter(trades_df_reset['Time'], trades_df_reset['Price'], marker='x', label=f'{seller}',
                            color='black')
            axes[0].set_ylabel("Trade Price")
            axes[0].set_title(f"Trade Prices Over Time")
            axes[0].legend()
            axes[0].grid()

            # Mark shock points with vertical lines
            # Mark shock events on price plot
            for shock_time in shock_times:
                axes[0].axvline(x=shock_time, color='red', linestyle='--', alpha=0.8)
                axes[1].axvline(x=shock_time, color='red', linestyle='--', alpha=0.8)

            # Filter entropy data to exclude early times before rolling window is full
            entropy_plot_df = trades_df_reset[trades_df_reset['Time_dt'] >= pd.Timedelta(seconds=time_window)]

            # Plot entropy
            axes[1].scatter(entropy_plot_df['Time'], entropy_plot_df['Entropy'], marker='x',
                            label=f'Entropy ({seller})', color='blue')
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Entropy")
            axes[1].set_title(f"Price Entropy Over Time")
            axes[1].legend()
            axes[1].grid()


            # Set x-axis limits
            axes[0].set_xlim(start_time, end_time)
            axes[1].set_xlim(start_time, end_time)

            plt.tight_layout()
            plt.show()

            # Define the shock time (here, assumed to occur at 300 seconds)
            shock_time = pd.Timedelta(seconds=300)

            # Filter the DataFrame for one minute (60 seconds) before and after the shock
            pre_shock = trades_df_reset[(trades_df_reset['Time_dt'] >= shock_time - pd.Timedelta(minutes=1)) &
                                        (trades_df_reset['Time_dt'] < shock_time)]
            post_shock = trades_df_reset[(trades_df_reset['Time_dt'] >= shock_time) &
                                         (trades_df_reset['Time_dt'] < shock_time + pd.Timedelta(minutes=1))]

            # Print out the average entropy for each period
            print("Average Entropy 1 minute pre shock:", pre_shock['Entropy'].mean())
            print("Average Entropy 1 minute post shock:", post_shock['Entropy'].mean())

