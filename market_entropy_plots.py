import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Functions _________________________

def calculate_price_entropy(data):
    # Calculate frequency of each bin
    freq = data.value_counts(normalize=True)
    # Calculate Shannon entropy
    return -np.sum(freq * np.log2(freq))

def calculate_entropy_from_distribution(distribution):
    """
    Calculate the Shannon entropy of a given price-quantity distribution.

    Parameters:
        distribution (list of tuples): Each tuple contains (price, quantity).

    Returns:
        float: The Shannon entropy of the distribution.
    """
    if not distribution:
        return 0  # No data means zero entropy

    # Extract quantities and normalize to probabilities
    quantities = np.array([q for _, q in distribution])
    total_quantity = quantities.sum()
    if total_quantity == 0:
        return 0  # Avoid division by zero
    probabilities = quantities / total_quantity

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def calculate_market_entropies(market_data):
    """
    Calculate the entropy for bids and asks from market data.

    Parameters:
        market_data (pd.DataFrame): DataFrame with columns:
            'Time', 'Bids', 'Best Bid', 'Asks', 'Best Ask'

    Returns:
        pd.DataFrame: Original DataFrame with added columns for 'Bid Entropy' and 'Ask Entropy'.
    """
    bid_entropies = []
    ask_entropies = []

    for _, row in market_data.iterrows():
        # Parse Bids and Asks as lists of tuples
        bids = eval(row.iloc[1])  # Convert string representation to list of tuples
        asks = eval(row.iloc[3])  # Convert string representation to list of tuples

        bid_entropy = calculate_entropy_from_distribution(bids)
        ask_entropy = calculate_entropy_from_distribution(asks)

        bid_entropies.append(bid_entropy)
        ask_entropies.append(ask_entropy)

    market_data['Bid Entropy'] = bid_entropies
    market_data['Ask Entropy'] = ask_entropies
    market_data['Bid Entropy Smoothed'] = market_data['Bid Entropy'].rolling(window=100, min_periods=10).mean()
    market_data['Ask Entropy Smoothed'] = market_data['Ask Entropy'].rolling(window=100, min_periods=10).mean()
    return market_data

#______________________________


if __name__ == "__main__":

    output = pd.DataFrame(columns=['buyer', 'n_traders', 'avg_entropy'])

    for i in range(1):

        from BSE import market_session

        # Set Up Simulation

        # Simulation parameters
        start_time = 0
        end_time = 60
        chart1_range = (1, 100)
        order_interval = 1

        supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
        demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [chart1_range], 'stepmode': 'fixed'}]
        order_sched = {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'periodic'}

        verbose = False
        dump_flags = {'dump_blotters': False, 'dump_lobs': True, 'dump_strats': False, 'dump_avgbals': True,
                      'dump_tape': True}

        # Loop over different numbers of traders
        num_traders = [100]

        for n in num_traders:

            # Define buyer types
            buyer_types = ['ZIP']
            for buyer_type in buyer_types:
                seller_type = 'ZIP'

                trial_id = f'test'

                # Specify sellers and buyers based on the current number of traders
                sellers_spec = [(seller_type, n)]
                buyers_spec = [(buyer_type, n)]
                traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

                # Run the market session
                market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)


                #Data Manipulation

                l2_data = trial_id + '_LOB_frames.csv'
                time_price_pairs_bids = []  # Initialize a list to hold (time, price) pairs for bids
                time_price_pairs_ask = []  # Initialize a list for asks
                best_prices = []

                df = pd.read_csv(l2_data)
                for i, row in df.iterrows():
                        time = row.iloc[0]  # Accessing by position in a pandas Series
                        bids = eval(row.iloc[1])
                        asks = eval(row.iloc[3])
                        bid_best = row.iloc[2]
                        ask_best = row.iloc[4]
                        # Append bid prices
                        for bid in bids:
                            price = bid[0]  # Get the bid price (first element of the tuple)
                            time_price_pairs_bids.append((time, price))  # Add tuple of (time, bid price)

                        # Append ask prices
                        for ask in asks:
                            price = ask[0]  # Get the ask price (first element of the tuple)
                            time_price_pairs_ask.append((time, price))  # Add tuple of (time, ask price)

                        if bid_best != '':
                            bid_best = float(bid_best)
                        else:
                            bid_best = None  # or '' for an empty string

                        if ask_best != '':
                            ask_best = float(ask_best)
                        else:
                            ask_best = None  # or '' for an empty string

                        best_prices.append((time, bid_best, ask_best))

                times = [bp[0] for bp in best_prices]
                best_bids = [bp[1] for bp in best_prices]
                best_asks = [bp[2] for bp in best_prices]

                # Convert to numpy arrays for easier plotting
                time_price_array_bids = np.array(time_price_pairs_bids)
                time_price_array_asks = np.array(time_price_pairs_ask)

                #Manipulating L2 data
                market_data = pd.read_csv(l2_data)
                # Calculate entropies
                market_data = calculate_market_entropies(market_data)
                headers = ['Time', 'Bids', 'Best Bid', 'Asks', 'Best Ask', 'Bid Entropy', 'Ask Entropy', 'Bid Entropy Smoothed', 'Ask Entropy Smoothed']
                market_data.columns = headers
                # Save the results to a new file
                output_file_path = f'{l2_data}_ENHANCED.csv'
                market_data.to_csv(output_file_path, index=False)

                trades_file = f'{trial_id}_tape.csv'
                trades_df = pd.read_csv(trades_file)
                trades_df.columns = ['Type', 'Time', 'Price']
                trades_df['id'] = range(len(trades_df))
                trades_df.set_index('id', inplace=True)
                trades_df = trades_df.sort_values(by="Time")
                trades_df['returns'] = trades_df['Price'].pct_change().dropna()
                bins = np.linspace((trades_df['returns'].min())*2, (trades_df['returns'].max())*2, 1000)
                trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
                trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=50).apply(lambda x: calculate_price_entropy(x.dropna()))
                mean_entropy = trades_df['Entropy'].mean()
                new_row = {'buyer': buyer_type, 'n_traders': n, 'avg_entropy': mean_entropy}
                output = pd.concat([output, pd.DataFrame([new_row])], ignore_index=True)
                output.to_csv('output.csv', index=True)








    # Plot bid and ask entropy only when they change
    fig, axs = plt.subplots(2, 1, figsize=(20,20))  # 1 row, 2 columns

    axs[0].plot(trades_df['Time'], trades_df['Price'], label='Trades', color='green', ls = '', marker='x')
    axs[0].set_title('Trade Prices Over Time')
    axs[0].set_xlabel('Trade Time')
    axs[0].set_ylabel('Trade Price')
    axs[0].set_xlim(start_time,end_time)
    axs[0].grid(True)

    # Second subplot (Trade Prices vs. Trade Times)
    axs[1].plot(trades_df['Time'], trades_df['Entropy'], label='Entropy', color='green', ls = '', marker='x')
    axs[1].set_title('Entropy Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Entropy')
    axs[1].set_xlim(start_time, end_time)
    axs[1].grid(True)


    # Show the plot
    plt.show()
