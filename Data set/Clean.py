import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the raw data
df = pd.read_csv('/home/anouarov/Pictures/csv_result-bone-marrow.csv')

# 2. Drop the useless and "cheating" columns
bad_columns = [
    'id', 'survival_time', 'time_to_aGvHD_III_IV', 
    'ANCrecovery', 'PLTrecovery', 'Relapse', 
    'aGvHDIIIIV', 'extcGvHD'
]
df = df.drop(columns=[col for col in bad_columns if col in df.columns], errors='ignore')

# 3. Fill missing values silently
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 4. Turn text into simple numbers (No True/False explosion)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Compress memory silently (Project Requirement)
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    elif df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')

# 6. Save the final, clean, simple dataset
df.to_csv('bmt_clean_simple.csv', index=False)
print("Done. Clean dataset saved as 'bmt_clean_simple.csv'.")