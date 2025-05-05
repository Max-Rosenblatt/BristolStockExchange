import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 24})

# Load the data
df = pd.read_csv('entropy_by_trader_types_variedN.csv')
df = df[df['buyer_type'] == 'GVWY']

# Set up the plot
plt.figure(figsize=(14, 7))

# Group by number of traders and buyer/seller types, then compute mean and standard error
grouped = df.groupby(['n_traders', 'buyer_type', 'seller_type'])['average_entropy']
summary = grouped.agg(['mean', 'sem']).reset_index()

# Create a unique label for each trader combination
summary['combo'] = summary['seller_type']


# Plot each combination
for combo in summary['combo'].unique():
    combo_data = summary[summary['combo'] == combo]
    plt.plot(
        combo_data['n_traders']*2,
        combo_data['mean'],
        label=combo,
        marker='x',
        linestyle='-'
    )

# Customize the plot
plt.ylim(0,10)
plt.xlabel('Number of Traders')
plt.ylabel('Average Entropy/Bits')
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig('GVWY_n_traders.png', dpi = 300)

# Show plot
plt.show()
