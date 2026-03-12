import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_clean_simple.csv'
if not os.path.exists(dataset_path):
    dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_dataset_processed.csv'

df = pd.read_csv(dataset_path)
out_dir = '/home/anouarov/.gemini/antigravity/brain/52e3efff-7e50-4609-a43f-067ef01bbfc1'
out_csv = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_dataset_log_transformed.csv'

col_name = 'CD34kgx10d6'
new_col_name = 'log_' + col_name

skew_before = df[col_name].skew()

# Apply log transformation
df[new_col_name] = np.log(df[col_name])

skew_after = df[new_col_name].skew()

print(f"--- SKEWNESS COMPARISON ---")
print(f"Original {col_name} Skewness: {skew_before:.4f}")
print(f"Log-Transformed {new_col_name} Skewness: {skew_after:.4f}")

# Plotting side by side histograms
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original Histogram
sns.histplot(df[col_name], kde=True, ax=axes[0], color='skyblue', bins=30)
axes[0].set_title(f"Original Distribution\n{col_name}", fontsize=14)
axes[0].set_xlabel("Values", fontsize=12)
axes[0].text(0.95, 0.95, f'Skewness: {skew_before:.4f}', 
             transform=axes[0].transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# Transformed Histogram
sns.histplot(df[new_col_name], kde=True, ax=axes[1], color='lightgreen', bins=30)
axes[1].set_title(f"Log-Transformed Distribution\n{new_col_name}", fontsize=14)
axes[1].set_xlabel("Values", fontsize=12)
axes[1].text(0.95, 0.95, f'Skewness: {skew_after:.4f}', 
             transform=axes[1].transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
fig_path = os.path.join(out_dir, "CD34kgx10d6_log_transformation.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

# Save the updated dataframe
df.to_csv(out_csv, index=False)
print(f"\nSaved transformation figure to: {fig_path}")
print(f"Saved updated dataset to: {out_csv}")
