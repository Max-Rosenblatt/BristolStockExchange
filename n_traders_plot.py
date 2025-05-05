import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the CSV files (same directory as the script)
folder = os.path.dirname(os.path.abspath(__file__))

# Data storage
crash_magnitudes = []
pre_shock_entropies = []
post_shock_entropies = []

# Regex to match filenames like 'entropy_results_50sims_50trad_ZIP_43.0%.csv'
pattern = re.compile(r'entropy_results.*_(\d+\.\d+)%\.csv')

# Iterate through files in the folder
for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        percentage = float(match.group(1))
        filepath = os.path.join(folder, filename)

        # Load CSV and extract entropy values
        df = pd.read_csv(filepath)

        # Assuming two columns: 'pre_shock_entropy', 'post_shock_entropy'
        pre_entropy = df['Pre_Shock_Entropy'].mean()
        post_entropy = df['Post_Shock_Entropy'].mean()

        # Store the data
        crash_magnitudes.append(percentage)
        pre_shock_entropies.append(pre_entropy)
        post_shock_entropies.append(post_entropy)

# Sort by crash magnitude
sorted_data = sorted(zip(crash_magnitudes, pre_shock_entropies, post_shock_entropies))
crash_magnitudes, pre_shock_entropies, post_shock_entropies = zip(*sorted_data)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(crash_magnitudes, pre_shock_entropies, label='Pre-shock Entropy', marker='o')
plt.plot(crash_magnitudes, post_shock_entropies, label='Post-shock Entropy', marker='s')
plt.xlabel('Crash Magnitude (%)')
plt.ylabel('Average Entropy')
plt.title('Entropy vs Crash Magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
