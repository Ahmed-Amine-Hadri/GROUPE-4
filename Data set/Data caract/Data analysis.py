import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


def get_statistical_profiling(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    profile = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Median': df[numeric_cols].median(),
        'Skewness': df[numeric_cols].apply(skew, nan_policy='omit'),
        'Kurtosis': df[numeric_cols].apply(kurtosis, nan_policy='omit')
    })
    return profile

def plot_univariate_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df, x='Recipientage', kde=True, ax=axes[0])
    axes[0].set_title('Recipientage Distribution')
    sns.histplot(data=df, x='CD34kgx10d6', kde=True, ax=axes[1])
    axes[1].set_title('CD34kgx10d6 Distribution')
    plt.tight_layout()
    plt.show()

def plot_bivariate_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x='survival_status', y='CD34kgx10d6', data=df, ax=axes[0])
    axes[0].set_title('CD34kgx10d6 vs Survival')
    sns.boxplot(x='survival_status', y='Recipientage', data=df, ax=axes[1])
    axes[1].set_title('Recipientage vs Survival')
    sns.boxplot(x='survival_status', y='Rbodymass', data=df, ax=axes[2])
    axes[2].set_title('Rbodymass vs Survival')
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(df):
    counts = df['survival_status'].value_counts()
    minority_class = counts.idxmin()
    minority_pct = (counts.min() / counts.sum()) * 100
    
    print(f"Minority class ({minority_class}) percentage: {minority_pct:.2f}%")
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Class Distribution (Survival Status)')
    plt.ylabel('Count')
    plt.xlabel('Survival Status')
    plt.show()

def plot_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr(method='spearman')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Spearman Correlation Heatmap')
    plt.show()
    
    if 'Recipientage' in corr_matrix.columns and 'Rbodymass' in corr_matrix.columns:
        corr_val = corr_matrix.loc['Recipientage', 'Rbodymass']
        if corr_val > 0.8:
            print(f"HIGHLIGHT: High correlation between Recipientage and Rbodymass (Spearman rho = {corr_val:.2f} > 0.8)")

def get_outliers_iqr(df, columns):
    outliers_dict = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outliers_dict[col] = outliers.tolist()
    return outliers_dict

df = pd.read_csv('/home/anouarov/Pictures/bmt_dataset_processed.csv')

print(get_statistical_profiling(df))
plot_bivariate_analysis(df)
analyze_class_distribution(df)
plot_correlation_matrix(df)
outliers = get_outliers_iqr(df, ['CD34kgx10d6', 'CD3dkgx10d8'])
print("Outliers count -> CD34kgx10d6:", len(outliers.get('CD34kgx10d6', [])))
print("Outliers count -> CD3dkgx10d8:", len(outliers.get('CD3dkgx10d8', [])))
