import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plotting style
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

# Load the CSV file
file_path = "HistoricalData_1739961641805.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2021-12-31')]

# Ensure 'Close/Last' is treated as numeric
df['Close/Last'] = pd.to_numeric(df['Close/Last'], errors='coerce')

# Sort values by date
df = df.sort_values(by='Date')

# Calculate log returns
df['returns'] = np.log(df['Close/Last'] / df['Close/Last'].shift(1))
returns = df['returns'].dropna()


# Plot histogram of log returns
plt.figure(figsize=(14, 7))
plt.hist(returns, bins=100, color='grey', edgecolor='black', alpha=0.7)
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('sp500_log_return_histogram.png', dpi=300)
plt.show()
