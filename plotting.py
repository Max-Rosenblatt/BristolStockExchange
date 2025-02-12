import pandas as pd

df = pd.read_csv('output.csv')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from BSE import market_session

plt.rcParams['figure.dpi'] = 300
font = {'size'   : 18}
plt.rc('font', **font)

# Create a scatter plot
plt.figure(figsize=(12, 8))

# Loop through each unique buyer type
for buyer_type in df['buyer'].unique():
    # Filter data for the current buyer type
    buyer_data = df[df['buyer'] == buyer_type]

    # Plot scatter for the current buyer type
    plt.scatter(buyer_data['n_traders'], buyer_data['avg_entropy'], label=buyer_type, s=100, marker = 'x', ls = '')

# Customize the plot
plt.xlabel('Number of Traders')
plt.ylabel('Average Session Entropy')
plt.title('Average Entropy vs Number of Traders for Different Buyer Types')
plt.legend(title="Buyer Types")
plt.grid(True)

# Show the plot
plt.show()