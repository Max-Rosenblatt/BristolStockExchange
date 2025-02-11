import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

n = 50
trial_id = '50_traders'

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

binsizes = np.linspace(10, 10000, 100, dtype = int)
data = []
for binsize in binsizes:
    # Bin returns and calculate entropy
    bins = np.linspace(-10, 10, binsize)
    trades_df['returns_binned'] = pd.cut(trades_df['returns'], bins=bins, labels=False)
    trades_df['Entropy'] = trades_df['returns_binned'].rolling(window=20).apply(
        lambda x: calculate_price_entropy(x.dropna()))

    """

    # Plotting Entropy and Price vs Time
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[1].plot(trades_df['Time'], trades_df['Entropy'], ls='', marker='x', color='navy')
    axs[0].plot(trades_df['Time'], trades_df['Price'], ls='', marker='x', color='black')

    axs[1].set_title(f"Entropy vs Time - {n} Traders - Number of Bins {binsize}")

    axs[1].set_xlabel(f"Time")

    axs[1].set_ylabel(f"Shannon Entropy/Bits")
    axs[0].set_ylabel(f"Price")

    print(trades_df['Entropy'].mean())
    data.append(trades_df['Entropy'].mean())

    plt.show()
    """
    data.append(trades_df['Entropy'].mean())
    print(trades_df['Entropy'].mean())


plt.plot(binsizes, data, marker = 'x', ls = '')
plt.show()
