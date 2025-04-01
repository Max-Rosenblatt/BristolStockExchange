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
rolling_std = df['Log_Return'].rolling(100).std()

# Define shock threshold (2 standard deviations)
shock_threshold = 3 * rolling_std

# Identify spikes and crashes (absolute log return exceeds threshold)
df['Shock'] = (df['Log_Return'].abs() > shock_threshold).astype(int)

# Ensure shocks are not counted twice by keeping only the first shock in a cluster
shock_indices = df.index[df['Shock'] == 1].tolist()
filtered_shock_dates = []
last_shock_date = None
cooldown_period = 30  # Minimum days between recorded shocks

for idx in shock_indices:
    shock_date = df.loc[idx, 'Date']

    if last_shock_date is None or (shock_date - last_shock_date).days > cooldown_period:
        filtered_shock_dates.append(shock_date)
        last_shock_date = shock_date  # Update last recorded shock date

shock_dates = filtered_shock_dates



# Function to calculate entropy
def calculate_price_entropy(data):
    freq = data.value_counts(normalize=True)
    return -np.sum(freq * np.log2(freq))


# Compute entropy using a rolling window of 100 days
bins = np.linspace(df['Log_Return'].min(), df['Log_Return'].max(), 2000)
df['returns_binned'] = pd.cut(df['Log_Return'], bins=bins, labels=False, duplicates='drop')
df['Entropy'] = df['returns_binned'].rolling(window=100).apply(
    lambda x: calculate_price_entropy(x.dropna()))

for shock_date in shock_dates:
    # Define time window (e.g., ±30 days)
    window_start = shock_date - pd.Timedelta(days=9)
    window_end = shock_date + pd.Timedelta(days=365)

    # Filter data for analysis
    df_window = df[(df['Date'] >= window_start) & (df['Date'] <= window_end)]

    # Get entropy just before and after the shock
    pre_shock_entropy = df_window[df_window['Date'] < shock_date]['Entropy'].dropna().iloc[-1] if not df_window[df_window['Date'] < shock_date]['Entropy'].dropna().empty else None
    post_shock_entropy = df_window[df_window['Date'] > shock_date]['Entropy'].dropna().iloc[0] if not df_window[df_window['Date'] > shock_date]['Entropy'].dropna().empty else None

    # Calculate entropy spike (if valid values exist)
    if pre_shock_entropy is not None and post_shock_entropy is not None:
        entropy_spike = post_shock_entropy - pre_shock_entropy
        print(f"Shock detected on {shock_date.date()}: Entropy spiked by {entropy_spike:.4f} bits")

    # Plot Closing Price and Entropy
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Closing Price (Primary Y-Axis)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price (USD)', color='black')
    ax1.plot(df_window['Date'], df_window['Close/Last'], color='black', label='Closing Price', marker='o', linestyle=' ')
    ax1.tick_params(axis='y', labelcolor='black')

    # Entropy (Secondary Y-Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Entropy (bits)', color='blue')
    ax2.plot(df_window['Date'], df_window['Entropy'], color='blue', label='Entropy', marker='o', linestyle=' ')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Mark the shock date
    ax1.axvline(shock_date, color='red', linestyle='--', label='Shock Event')

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Formatting
    plt.title(f'Market Shock & Entropy Spike on {shock_date.date()}')
    plt.grid()
    plt.show()
