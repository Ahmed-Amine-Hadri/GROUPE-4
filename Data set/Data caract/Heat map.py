import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_correlation_multicollinearity(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method='spearman')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Spearman Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    print("Highly Correlated Feature Pairs (|rho| > 0.75):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) > 0.75:
                print(f"- {col1} & {col2}: {corr_val:.2f}")
                
    if 'Recipientage' in corr_matrix.columns and 'Rbodymass' in corr_matrix.columns:
        spec_corr = corr_matrix.loc['Recipientage', 'Rbodymass']
        if abs(spec_corr) > 0.8:
            print("\n" + "!" * 80)
            print(f"WARNING: HIGH MULTICOLLINEARITY DETECTED (|rho| = {abs(spec_corr):.2f} > 0.8)")
            print("Recipientage and Rbodymass are highly correlated.")
            print("Consider using PCA or dropping one of these features to prevent model instability.")
            print("!" * 80 + "\n")

if __name__ == "__main__":
    try:
        df = pd.read_csv('/home/anouarov/Pictures/bmt_dataset_processed.csv')
        analyze_correlation_multicollinearity(df)
    except FileNotFoundError:
        pass
