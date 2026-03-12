import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> list:
    outlier_indices = set()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.update(indices)
    return list(outlier_indices)

def detect_outliers_zscore(df: pd.DataFrame, columns: list) -> list:
    outlier_indices = set()
    for col in columns:
        if col in df.columns:
            valid_data = df[col].dropna()
            z_scores = np.abs(zscore(valid_data))
            indices = valid_data[z_scores > 3].index
            outlier_indices.update(indices)
    return list(outlier_indices)

def clip_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_clipped = df.copy()
    for col in columns:
        if col in df_clipped.columns:
            lower_percentile = df_clipped[col].quantile(0.01)
            upper_percentile = df_clipped[col].quantile(0.99)
            df_clipped[col] = df_clipped[col].clip(lower=lower_percentile, upper=upper_percentile)
    return df_clipped

def apply_robust_scaler(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_scaled = df.copy()
    cols_to_scale = [col for col in columns if col in df_scaled.columns]
    
    if cols_to_scale:
        scaler = RobustScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
        
    return df_scaled

def plot_outliers_before_after(df_original, df_cleaned, column):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(y=df_original[column], ax=axes[0], color='lightcoral')
    axes[0].set_title(f'{column} (Before: Raw Data)')
    axes[0].set_ylabel('Cell Dosage')
    
    sns.boxplot(y=df_cleaned[column], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'{column} (After: Clipped Data)')
    axes[1].set_ylabel('Cell Dosage')
    
    plt.tight_layout()
    plt.show()

df = pd.read_csv('/home/anouarov/Pictures/bmt_dataset_processed.csv')
target_cols = ['CD34kgx10d6', 'CD3dkgx10d8']

iqr_outliers = detect_outliers_iqr(df, target_cols)
print(len(iqr_outliers))

zscore_outliers = detect_outliers_zscore(df, target_cols)
print(len(zscore_outliers))

df_clipped = clip_outliers(df, target_cols)
print(df_clipped[target_cols].max())

df_scaled = apply_robust_scaler(df, target_cols)
print(df_scaled[target_cols].head())

plot_outliers_before_after(df, df_clipped, 'CD34kgx10d6')
plot_outliers_before_after(df, df_clipped, 'CD3dkgx10d8')