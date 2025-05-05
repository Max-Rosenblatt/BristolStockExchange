import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Font settings
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})

# Load the data
df = pd.read_csv('advancedshock_entropy_results_100sims_50trad_allCombinations.csv')

ordered_types = ['GVWY', 'SHVR', 'ZIC', 'ZIP']
buyer_types = ordered_types
seller_types = ordered_types

# Create a grid of subplots: buyers on rows, sellers on columns
fig, axes = plt.subplots(len(buyer_types), len(seller_types), figsize=(18, 14), sharex=True, sharey=True)

# Ensure axes is a 2D array
if len(buyer_types) == 1 and len(seller_types) == 1:
    axes = np.array([[axes]])
elif len(buyer_types) == 1 or len(seller_types) == 1:
    axes = axes.reshape(len(buyer_types), len(seller_types))

# Fill each subplot
for row_idx, buyer in enumerate(buyer_types):
    for col_idx, seller in enumerate(seller_types):
        ax = axes[row_idx, col_idx]

        subset = df[(df['buyer_type'] == buyer) & (df['seller_type'] == seller)]

        if not subset.empty:
            avg_pre = subset['average_entropy_pre_shock'].mean()
            avg_post = subset['average_entropy_post_shock'].mean()

            bars = ax.bar(['Pre-Shock', 'Post-Shock'], [avg_pre, avg_post],
                          color=['blue', 'red'], alpha = 0.4)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., (height-1)/2,
                        f'{height:.2f}',
                        ha='center', va='bottom')


            pct_change = ((avg_post - avg_pre) / avg_pre) * 100
            print(f'Buyer: {buyer} | Seller: {seller} | pct change: {pct_change:.2f}%')

            ax.set_ylim(0, 10)

        else:
            ax.set_visible(False)

        # Label buyers along the left
        if col_idx == len(seller_types) - 1:
            ax.set_ylabel(f'{buyer}', rotation=90, labelpad=20, va='center', ha='left')
            ax.yaxis.set_label_position("right")

        # Label sellers along the bottom
        if row_idx == len(buyer_types) - 1:
            ax.set_xlabel(f'{seller}', rotation=0, labelpad=10)

        # Remove axis ticks
        ax.set_xticks([])
        if col_idx == 0:
            # Add y-axis ticks and label for the leftmost plots
            ax.set_yticks([0, 2, 4, 6, 8, 10])  # or use `ax.set_yticks(np.linspace(0, 10, 6))`
            ax.set_yticklabels([str(tick) for tick in [0, 2, 4, 6, 8, 10]])

# Add shared labels
fig.text(0.95, 0.5, 'Buyer Type', va='center', rotation='vertical')
fig.text(0.04, 0.5, 'Entropy/Bits', va='center', rotation='vertical')
fig.text(0.5, 0.05, 'Seller Type', ha='center')

plt.savefig('shock_entropy_matrix.png', dpi = 300)

plt.show()
