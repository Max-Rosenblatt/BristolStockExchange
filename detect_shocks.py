import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
file_path = "HistoricalData_1739961641805.csv"
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Ensure 'Close/Last' is numeric
df['Close/Last'] = pd.to_numeric(df['Close/Last'], errors='coerce')

# Sort by date
df = df.sort_values(by='Date')

# Compute log returns
df['Log_Return'] = np.log(df['Close/Last'] / df['Close/Last'].shift(1))

# Compute rolling standard deviation over a 100-day window
rolling_std = df['Log_Return'].rolling(14).std()

# Define shock threshold (2 standard deviations)
shock_threshold = 3 * rolling_std

# Identify spikes and crashes (absolute log return exceeds threshold)
df['Shock'] = (df['Log_Return'].abs() > shock_threshold).astype(int)

# Extract shock event dates
shock_dates = df.loc[df['Shock'] == 1, 'Date'].tolist()


# Function to calculate entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


# Compute entropy using a rolling window of 100 days
bins = np.linspace(df['Log_Return'].min() * 2, df['Log_Return'].max() * 2, 2000)
df['returns_binned'] = pd.cut(df['Log_Return'], bins=bins, labels=False, duplicates='drop')
df['Entropy'] = df['returns_binned'].rolling(window=100).apply(
    lambda x: calculate_price_entropy(x.dropna()))

# Plot entropy + price around each shock event
for shock_date in shock_dates:

    # Define time window (e.g., ±14 days)
    window_start = shock_date - pd.Timedelta(days=100)
    window_end = shock_date + pd.Timedelta(days=100)

    # Filter data for plotting
    df_window = df[(df['Date'] >= window_start) & (df['Date'] <= window_end)]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Closing Price (Primary Y-Axis)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price (USD)', color='black')
    ax1.plot(df_window['Date'], df_window['Close/Last'], color='black', label='Closing Price', marker='o',
             linestyle=' ')
    ax1.tick_params(axis='y', labelcolor='black')

    # Add secondary y-axis for entropy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Entropy (bits)', color='blue')
    ax2.plot(df_window['Date'], df_window['Entropy'], color='blue', label='Entropy', marker='o', linestyle=' ')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Mark the shock event
    ax1.axvline(shock_date, color='red', linestyle='--', label='Shock Event')

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Formatting
    plt.title(f'Market Shock Analysis on {shock_date.date()}')
    plt.grid()
    plt.show()
