import joblib
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

def main():
    project_root = Path(r'C:\Users\pc\Documents\Git\Gittt\GROUPE-4')
    data_path = project_root / 'src' / 'final_dataset.csv'    
    models_dir = project_root / 'models'
    
    df_raw = pd.read_csv(data_path)
    target_col = 'survival_status'
    y = df_raw[target_col]
    X_raw = df_raw.drop(columns=[target_col])
    
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)
    
    models_info = {
        'XGBoost': models_dir / 'xgboost_model.pkl',
        'SVM': models_dir / 'modele_svm_bmt.pkl',
        'Random Forest': models_dir / 'rf_model.pkl',
        'LightGBM': models_dir / 'lgbm_model.pkl'
    }
    
    results = []
    
    for name, path in models_info.items():
        if not path.exists():
            continue
            
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred 
            
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc = 0.0
            
        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1-Score': f1,
            'Precision': prec,
            'Recall': rec,
            'ROC-AUC': roc
        })
        
    if not results:
        return
        
    df_results = pd.DataFrame(results)
    
    print("| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |")
    print("|---|---|---|---|---|---|")
    for _, row in df_results.iterrows():
        print(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['F1-Score']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['ROC-AUC']:.4f} |")
    
    df_melt = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model')
    plt.title('Final Model Comparison')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(models_dir / 'final_model_comparison.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()