import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

plt.rcParams['date.converter'] = 'concise'

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Load the CSV file
file_path = "data/HistoricalData_1739961641805.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Ensure 'Close/Last' is treated as numeric
df['Close/Last'] = pd.to_numeric(df['Close/Last'], errors='coerce')

# Sort values by date
df = df.sort_values(by='Date')

# Calculate percentage changes
df['returns'] = np.log(df['Close/Last'] /df['Close/Last'].shift(1)).dropna()

bins = np.linspace(df['returns'].min()*1.1, df['returns'].max()*1.1, 2000)
df['returns_binned'] = pd.cut(df['returns'], bins=bins, labels=False, duplicates='drop')
df['Entropy'] = df['returns_binned'].rolling(window=90).apply(
    lambda x: calculate_price_entropy(x.dropna()))


# Save to CSV
df.to_csv("sp500_entropy_full.csv", index=False)


# Calculate 90-day rolling volatility (standard deviation of returns)
df['Volatility'] = df['returns'].rolling(window=90).std()

# Create two vertically stacked subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# --- Bottom subplot: Volatility ---
ax1.plot(df['Date'], df['Close/Last'], linestyle=' ', color='black', marker = '.', markersize = 2)
ax1.set_ylabel('Price / USD')


# --- Top subplot: Closing Price and Entropy ---
ax2.set_ylabel('Volatility')
ax2.plot(df['Date'], df['Volatility'], linestyle='-', color='blue', label='Volatility')
ax2b = ax2.twinx()
ax2b.set_ylabel('Entropy / Bits')
ax2b.plot(df['Date'], df['Entropy'], linestyle='-', color='red', label='Entropy')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.savefig('sp500_price_entropy_volatility.png', dpi=300)
plt.show()