import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate price entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))

# Load the CSV file
file_path = "HistoricalData_1739961641805.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Ensure 'Close/Last' is treated as numeric
df['Close/Last'] = pd.to_numeric(df['Close/Last'], errors='coerce')

# Sort values by date
df = df.sort_values(by='Date')

# Calculate percentage changes
df['Pct_Change'] = df['Close/Last'].pct_change()

bins = np.linspace(df['Pct_Change'].min()*2, df['Pct_Change'].max()*2, 2000)
df['returns_binned'] = pd.cut(df['Pct_Change'], bins=bins, labels=False, duplicates='drop')
df['Entropy'] = df['returns_binned'].rolling(window=100).apply(
    lambda x: calculate_price_entropy(x.dropna()))

# Plot the closing price and entropy over time
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Price Change vs Previous Day/%')
#ax1.plot(df['Date'], df['Pct_Change'], marker='x', linestyle=' ', color='black', label='Closing Price', markersize = 2)
#ax1.plot(df['Date'], df['Close/Last'], marker='x', linestyle=' ', color='black', label='Closing Price', markersize = 2)
ax1.bar(df['Date'], df['Pct_Change'], color='black', alpha=0.6, label='Percentage Change')
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.set_ylabel('Entropy/Bits')
ax2.plot(df['Date'], df['Entropy'], marker='x', linestyle=' ', color='blue', label='Entropy', markersize = 2)
ax2.tick_params(axis='y')
ax2.legend(loc='upper right')

plt.title('S&P500 Price and Market Entropy')
plt.grid()
plt.show()
