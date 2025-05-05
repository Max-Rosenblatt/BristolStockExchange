import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.pylabtools import figsize

data = pd.read_csv('entropy_summary.csv')
fig = plt.figure()
plt.plot(data['Run'], data['PostEntropy']-data['PreEntropy'], ls = '', marker = 'x')
plt.show()