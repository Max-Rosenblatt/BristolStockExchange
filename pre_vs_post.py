import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BSE import market_session
import math

# Global state variables
market_state = {
    'price': 100,  # Initial price level
    'drift': 1e-5,  # Small upward drift to reflect overall growth
    'crash_active': False,
    'crash_start_time': None,
}

shock_times = []  # Store times when market crashes



# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


def schedule_offsetfn(t):
    global market_state, shock_times
    # Update volatility relative to current price (scale with price)
    market_state['volatility'] = 5e-6 * market_state['price']

    # Standard price update: Brownian motion with drift
    var = np.random.normal(loc=market_state['drift'], scale=market_state['volatility'])
    market_state['price'] += var

    # Occasional small jump to simulate microstructure noise (random event)
    if np.random.random() < 0.001:  # low probability for minor jump
        small_jump = np.random.normal(0, 0.01 * market_state['price'])
        market_state['price'] += small_jump

    # Trigger a major crash at a fixed time (t = 300 seconds)
    if (not market_state.get('crash_active', False) and
            market_state.get('crash_start_time') is None and
            math.floor(t) == 300):
        market_state['crash_active'] = True
        market_state['crash_start_time'] = t
        market_state['crash_price'] = market_state['price']
        # Use a random factor to simulate variability in crash severity (e.g., 20-40% drop)
        crash_jump_factor = np.random.uniform(0.6, 0.8)
        market_state['price'] = market_state['crash_price'] * crash_jump_factor
        market_state['negative_drift_duration'] = 60  # seconds of extra downward pressure (panic selling)
        shock_times.append(t)  # record the shock time for plotting
        print(
            f"Market crash at t={t}, price dropped to {market_state['price']:.2f} with factor {crash_jump_factor:.2f}")

    # During the negative drift phase (panic selling), apply additional downward pressure
    if market_state.get('negative_drift_duration', 0) > 0:
        negative_drift = -0.5  # extra downward pressure
        market_state['price'] += negative_drift * market_state['volatility']
        market_state['negative_drift_duration'] -= 1

    # After the negative drift phase, if a crash is active, simulate recovery with high volatility
    if market_state.get('crash_active', False) and market_state.get('negative_drift_duration', 0) <= 0:
        # Spike volatility to reflect market panic during the recovery phase
        spike_volatility = 2 * market_state['volatility']
        recovery_noise = np.random.normal(loc=market_state['drift'], scale=spike_volatility)
        market_state['price'] += recovery_noise

        # Exponential recovery toward the pre-crash price
        recovery_strength = 0.005
        target_price = market_state['crash_price']
        market_state['price'] += (target_price - market_state['price']) * recovery_strength

        # When price nears the pre-crash level, end the crash period
        if market_state['price'] >= target_price * 0.98:
            market_state['crash_active'] = False
            print(f"Market recovery complete at t={t}, price back to {market_state['price']:.2f}")

    return int(round(market_state['price'], 0))


rangeS = (100, 100, schedule_offsetfn)
rangeD = rangeS

# Simulation parameters
start_time = 0
end_time = 600
order_interval = 1

supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeS], 'stepmode': 'jittered'}]
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [rangeD], 'stepmode': 'jittered'}]

order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-fixed'}

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
            trial_id = f'test'
            buyers_spec = [('GVWY',25),('SHVR',25),('ZIC',25),('ZIP',25)]
            sellers_spec = buyers_spec
            traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

            # Fixed parameters for bins and time window
            n_bins = 2000
            time_window = 60

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


import seaborn as sns
plt.figure(figsize=(10,5))
sns.histplot(trades_df['returns'].dropna(), bins=50, kde=True)
plt.title("Distribution of Returns")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.show()

