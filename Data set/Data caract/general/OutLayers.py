import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_clean_simple.csv'
if not os.path.exists(dataset_path):
    dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_dataset_processed.csv'

df = pd.read_csv(dataset_path)
columns = ['Donorage', 'Recipientage', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass']
out_dir = '/home/anouarov/.gemini/antigravity/brain/52e3efff-7e50-4609-a43f-067ef01bbfc1'

print("--- OUTLIER STATISTICS ---")
total_rows = len(df)
results = []
for col in columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    percentage = (outlier_count / total_rows) * 100
    
    results.append({
        'Variable': col,
        'Lower Bound': round(lower_bound, 4),
        'Upper Bound': round(upper_bound, 4),
        'Outliers Count': outlier_count,
        'Percentage (%)': round(percentage, 2)
    })
    
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Plotting
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue', fliersize=5)
    axes[i].set_title(f"Boxplot of {col}")
    axes[i].set_ylabel("Value")

plt.tight_layout()
save_path = os.path.join(out_dir, "combined_boxplots.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\\nSaved figure to {save_path}")
