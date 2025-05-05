import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})

# Load data
df = pd.read_csv("entropy_by_trader_types_50_v2.csv")

# Pivot to form matrix
# Group by buyer and seller type
grouped = df.groupby(['buyer_type', 'seller_type'])

# Mean entropy across all runs
entropy_matrix = grouped['average_entropy'].mean().unstack()

# Standard error across all runs (i.e. SEM of the means)
err_matrix = grouped['average_entropy'].sem().unstack()

# Plot
plt.figure(figsize=(12, 10))
heatmap = plt.imshow(entropy_matrix, cmap='viridis', aspect='auto')

# Axis labels
plt.xticks(ticks=range(len(entropy_matrix.columns)), labels=entropy_matrix.columns)
plt.yticks(ticks=range(len(entropy_matrix.index)), labels=entropy_matrix.index)

# Colorbar
plt.colorbar(heatmap, label='Average Entropy/bits')

# Annotate each cell with its value
for i in range(len(entropy_matrix.index)):
    for j in range(len(entropy_matrix.columns)):
        value = entropy_matrix.iloc[i, j]
        err_value = err_matrix.iloc[i, j]
        if not pd.isna(value):
            plt.text(j, i, f"{value:.2f} ± {err_value:.2f}", ha='center', va='center', color='white' if value < 4.5 else 'black')

# Titles


plt.xlabel("Seller Type")
plt.ylabel("Buyer Type")

plt.tight_layout()
plt.savefig('entropy_matrix.png', dpi=300)
plt.show()


trade_count_matrix = df.pivot_table(
    index='buyer_type',
    columns='seller_type',
    values='num_trades',  # <- replace with actual column name
    aggfunc='sum'
)

print(trade_count_matrix)
