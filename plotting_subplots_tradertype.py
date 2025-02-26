import matplotlib.pyplot as plt
import pandas as pd

# Set up figures and axes (4x4 grid for each plot type)
fig_prices, axes_prices = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)
fig_entropy, axes_entropy = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)

# Define colors for traders
colors = ['red', 'blue', 'green', 'purple']
buyer_types = ['ZIP', 'ZIC', 'GVWY', 'SHVR']
seller_types = buyer_types

# Loop through trader combinations
for i, seller in enumerate(seller_types):
    for j, buyer in enumerate(buyer_types):
        trial_id = f'40_{buyer}B_{seller}S'
        trades_file = f'{trial_id}_stats.csv'

        # Load trade data
        trades_df = pd.read_csv(trades_file)

        # Select appropriate subplots
        ax_price = axes_prices[i, j]
        ax_entropy = axes_entropy[i, j]

        # Plot trade prices
        ax_price.scatter(trades_df['Time'], trades_df['Price'], marker='x', color=colors[j], alpha=0.6)
        ax_price.grid(True)

        # Plot entropy
        ax_entropy.scatter(trades_df['Time'], trades_df['Entropy'], marker='x', color=colors[i], alpha=0.6)
        ax_entropy.grid(True)


        # Add row labels for seller types (leftmost column)
        if j == 0:
            ax_price.set_ylabel(seller, fontsize=10, fontweight='bold')
            ax_entropy.set_ylabel(seller, fontsize=10, fontweight='bold')

        # Add column labels for buyer types (topmost row)
        if i == 0:
            ax_price.set_title(buyer, fontsize=10, fontweight='bold')
            ax_entropy.set_title(buyer, fontsize=10, fontweight='bold')

# Add overall figure labels
fig_prices.supxlabel("Buyer Type", fontsize=12, fontweight="bold")
fig_prices.supylabel("Seller Type", fontsize=12, fontweight="bold")

fig_entropy.supxlabel("Buyer Type", fontsize=12, fontweight="bold")
fig_entropy.supylabel("Seller Type", fontsize=12, fontweight="bold")


# Add global titles
fig_prices.suptitle("Trade Prices Over Time", fontsize=14)
fig_entropy.suptitle("Price Entropy Over Time", fontsize=14)

# Adjust layout
fig_prices.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
fig_entropy.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# Show plots
plt.show()


