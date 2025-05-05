import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

plt.rcParams['date.converter'] = 'concise'

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Load and prepare data
file_path = "HistoricalData_1739961641805.csv"
df = pd.read_csv(file_path)

# Convert and clean data
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Close/Last'] = pd.to_numeric(df['Close/Last'])
df = df.sort_values(by='Date')

# Calculate percentage changes and entropy (commented out as not used in final plot)
df['Pct_Change'] = df['Close/Last'].pct_change()
bins = np.linspace(df['Pct_Change'].min()*1.1, df['Pct_Change'].max()*1.1, 2000)
df['returns_binned'] = pd.cut(df['Pct_Change'], bins=bins, labels=False, duplicates='drop')
df['Entropy'] = df['returns_binned'].rolling(window=100).apply(
    lambda x: calculate_price_entropy(x.dropna()))

# Create figure with adjusted layout
fig, ax = plt.subplots(figsize=(14, 7))

# Plot closing prices
ax.plot(df['Date'], df['Close/Last'],
        marker='x', linestyle=' ',
        color='black',
        label='S&P500 Closing Price',
        markersize=2)


# Set labels with increased padding
ax.set_xlabel('Date', labelpad=15)
ax.set_ylabel('Price/USD', labelpad=25)  # Increased from 15 to 25

# Set limits and legend
ax.set_xlim(pd.Timestamp('2019-01-01'), pd.Timestamp('2021-12-31'))
ax.set_ylim(1500, 5000)

# Get current y-ticks and remove the bottom one
current_yticks = ax.get_yticks()
new_yticks = [tick for tick in current_yticks if tick > 1500]  # Exclude bottom tick (1500)

# Set new y-ticks
ax.set_yticks(new_yticks)


plt.tight_layout()

# Save and show
plt.savefig('sp500_crash_basic.png', dpi=300)
plt.show()