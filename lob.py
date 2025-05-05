import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 24})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Define spacing and bar width
bar_width = 0.25
price_step = 0.25

# Generate touching bid/ask levels
best_bid = 99.75
best_ask = best_bid + bar_width  # Ensures bars touch

# Generate prices
bid_prices = np.array([best_bid - i * price_step for i in range(8)])
ask_prices = np.array([best_ask + i * price_step for i in range(8)])

# Example volumes
bid_volumes = [80, 100, 120, 150, 180, 200, 220, 250]
ask_volumes = [110, 120, 140, 160, 170, 180, 200, 220]

# Offset positions for bars
bid_positions = bid_prices - bar_width / 2
ask_positions = ask_prices + bar_width / 2

# Calculate mid-price and spread
mid_price = (best_bid + best_ask) / 2
spread = best_ask - best_bid

# Plot
plt.figure(figsize=(14, 6))
plt.bar(bid_positions, bid_volumes, width=bar_width, color='green', label='Bids', alpha=0.5)
plt.bar(ask_positions, ask_volumes, width=bar_width, color='red', label='Asks', alpha=0.5)
plt.axvspan(xmin=best_ask, xmax=best_bid, color='grey', alpha=0.2, hatch = '\\', label='Spread')

# Mark mid-price and spread
plt.axvline(mid_price, color='blue', linestyle='--', linewidth=1.5, label='Mid-price')


# Labels and styling
plt.xlabel('Price')
plt.ylabel('Volume')
plt.tight_layout()
plt.savefig('lob.png', dpi = 300)
plt.show()
