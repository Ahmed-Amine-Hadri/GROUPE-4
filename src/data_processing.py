import pandas as pd
import numpy as np
import logging
import sys
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.utils import get_logger

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
logger = get_logger(__name__)

def analyze_distributions(df: pd.DataFrame):
    logger.info("Analyzing feature distributions...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[num_cols].skew().sort_values(ascending=False)
    
    high_skew = skewness[abs(skewness) > 0.75]
    for col, val in high_skew.items():
        logger.info(f"Targeting '{col}' for transformation: Skewness is {val:.2f}")
    
    return high_skew.index.tolist()

def analyze_correlations(df: pd.DataFrame, target_col: str = 'survival_status'):
    logger.info("Exploiting heatmap for feature redundancy...")
    corr_matrix = df.corr(method='spearman').abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    
    if to_drop:
        logger.info(f"Heatmap analysis identified {len(to_drop)} redundant features: {to_drop}")
    
    return corr_matrix, to_drop

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Task: Optimizing memory usage...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64': df[col] = df[col].astype('float32')
        elif col_type == 'int64': df[col] = df[col].astype('int32')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Task: Cleaning data and handling missing values...")
    bad_columns = ['id', 'survival_time', 'time_to_aGvHD_III_IV', 'Relapse', 'time', 'date']
    df = df.drop(columns=[c for c in bad_columns if c in df.columns], errors='ignore')
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return optimize_memory(df)

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling outliers via clipping...")
    target_cols = ['CD34kgx10d6', 'CD3dkgx10d8', 'WBCx10d8', 'MNCkgx10d8', 'RNCdkgx10d8']
    for col in [c for c in target_cols if c in df.columns]:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def apply_log_transformations(df: pd.DataFrame, skewed_cols: list) -> pd.DataFrame:
    logger.info("Applying log transformations based on distribution analysis...")
    for col in skewed_cols:
        if col in df.columns and col != 'survival_status':
            new_col = f'log_{col}'
            df[new_col] = np.log(df[col].replace(0, np.nan))
            df[new_col] = df[new_col].fillna(df[new_col].median())
            df = df.drop(columns=[col])
    return df

def reduce_multicollinearity(df: pd.DataFrame, corr_matrix, target_col: str = 'survival_status') -> pd.DataFrame:
    logger.info("Removing redundant features to optimize model performance...")
    target_corr = corr_matrix[target_col].fillna(0).sort_values(ascending=False)
    features = target_corr.index.drop(target_col).tolist()
    
    keep = []
    for feat in features:
        if not any(corr_matrix.loc[feat, k] > 0.80 for k in keep):
            keep.append(feat)
    
    return df[keep + [target_col]]

def prepare_and_augment_data(df: pd.DataFrame, processed_dir: Path):
    logger.info("--- STEP 1 : Loading Data for Augmentation ---")
    logger.info(f"Original Dataset: {df.shape[0]} patients, {df.shape[1]} columns.")

    X = df.drop(columns=['survival_status'])
    y = df['survival_status']

    logger.info("--- STEP 2 : 90/10 Split (Holdout) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"Train Set (before augmentation): {X_train.shape[0]} patients.")
    logger.info(f"Test Set (preserved): {X_test.shape[0]} patients.")

    logger.info("--- STEP 3 : Massive Augmentation (SMOTE) ---")
    strategy = {0: 200, 1: 200} 
    
    try:
        smote = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=5)
        X_train_augmented, y_train_augmented = smote.fit_resample(X_train, y_train)
    except ValueError:
        logger.info("Strict 200/200 rule not applicable. Defaulting to perfectly balanced classes.")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_augmented, y_train_augmented = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Train Set (AFTER augmentation): {X_train_augmented.shape[0]} patients.")
    logger.info(f"New distribution:\n{y_train_augmented.value_counts().to_string()}")

    logger.info("--- STEP 4 : Saving new Datasets ---")
    df_train_augmented = pd.concat([X_train_augmented, y_train_augmented], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    train_filename = processed_dir / 'augmented_train_dataset.csv'
    test_filename = processed_dir / 'holdout_test_dataset.csv'
    
    df_train_augmented.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)

    logger.info("Files successfully generated.")
    logger.info(f"Training set saved to: {train_filename.name}")
    logger.info(f"Final Test set saved to: {test_filename.name}")

def main():
    RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'csv_result-bone-marrow.csv'
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_PATH}")
        return
        
    logger.info("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    
    df = clean_data(df)
    df = handle_outliers(df)
    
    skewed_cols = analyze_distributions(df)
    df = apply_log_transformations(df, skewed_cols)
    
    corr_matrix, to_drop = analyze_correlations(df)
    df = reduce_multicollinearity(df, corr_matrix)
    
    # Passing directly to augmentation without saving intermediate baseline data
    prepare_and_augment_data(df, PROCESSED_DIR)
    
    logger.info("Data Processing Pipeline completed successfully.")

if __name__ == "__main__":
    main()
