import pandas as pd
import matplotlib.pyplot as plt

# Replace 'data.csv' with your actual file path
df = pd.read_csv('alphagrad/src/alphagrad/approx/nn_res.csv')

# Plotting on separate subplots due to differing orders of magnitude
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

df['entropy'].plot(ax=axes[0], color='tab:blue', title='Entropy', ylabel='Value')
df['best'].plot(ax=axes[1], color='tab:green', title='Best Score', ylabel='Value')
df['mean'].plot(ax=axes[2], color='tab:orange', title='Mean Score', ylabel='Value')

plt.xlabel('Episode / Step')
plt.tight_layout()
plt.show()
plt.savefig()