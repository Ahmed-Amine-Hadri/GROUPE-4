import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('/home/anouarov/Documents/GitHub/GROUPE-4/bmt_clean_simple.csv')

# Identify numerical variables (we will check all, but note which ones are continuous vs categorical)
# Often in such datasets, columns with many unique values are continuous
continuous_vars = [col for col in df.columns if df[col].nunique() > 10 and pd.api.types.is_numeric_dtype(df[col])]
print("Variables identified as continuous (>10 unique values):", continuous_vars)

results = []
for col in continuous_vars:
    s = df[col].skew()
    min_val = df[col].min()
    max_val = df[col].max()
    zeros = (df[col] == 0).sum()
    negatives = (df[col] < 0).sum()
    results.append({
        'Variable': col, 
        'Skewness': round(s, 4), 
        'Min': round(min_val, 4), 
        'Max': round(max_val, 4), 
        'Zeros': zeros, 
        'Negatives': negatives
    })

results_df = pd.DataFrame(results)
print("\n--- SKEWNESS RESULTS ---")
print(results_df.to_string(index=False))

acceptable = results_df[(results_df['Skewness'] >= -1) & (results_df['Skewness'] <= 1)]
significant = results_df[(results_df['Skewness'] < -1) | (results_df['Skewness'] > 1)]

print("\n--- ACCEPTABLE (-1 to +1) ---")
print([str(x) for x in acceptable['Variable'].tolist()])

print("\n--- SIGNIFICANT (<-1 or >+1) ---")
print([str(x) for x in significant['Variable'].tolist()])
