import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Suppress the seaborn warning for distplot
import warnings
warnings.filterwarnings('ignore')

dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_clean_simple.csv'
if not os.path.exists(dataset_path):
    # Try bmt_dataset_processed.csv if user renamed it
    dataset_path = '/home/anouarov/Documents/GitHub/GROUPE-4/bmt_dataset_processed.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found in {os.path.dirname(dataset_path)}", file=sys.stderr)
        sys.exit(1)

df = pd.read_csv(dataset_path)
columns = ['Donorage', 'Recipientage', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass']
out_dir = '/home/anouarov/.gemini/antigravity/brain/52e3efff-7e50-4609-a43f-067ef01bbfc1'

print("Using dataset:", dataset_path)
print("Generating diagrams in:", out_dir)

sns.set_style("whitegrid")

for col in columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in dataset.", file=sys.stderr)
        continue
        
    skewness = df[col].skew()
    
    # Create a figure with two subplots: Boxplot on top, Histogram on bottom
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 6))
    
    # Create the boxplot
    sns.boxplot(x=df[col], ax=ax_box, color='skyblue', fliersize=5)
    ax_box.set(xlabel='')
    ax_box.set_title(f"Distribution and Outlier Analysis for {col}", fontsize=14, pad=15)
    
    # Create the histogram with KDE
    sns.histplot(df[col], kde=True, ax=ax_hist, color='skyblue', bins=30, line_kws={'linewidth': 2})
    
    ax_hist.set_xlabel(f"{col} Values", fontsize=12)
    ax_hist.set_ylabel("Frequency / Density", fontsize=12)
    
    # Add skewness annotation in the histogram
    # To place it in the upper right corner relative to the axes
    ax_hist.text(0.95, 0.95, f'Skewness: {skewness:.4f}', 
                 transform=ax_hist.transAxes, 
                 fontsize=12, verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                 
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"{col}_dist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
print("Successfully generated all visual plots.")
