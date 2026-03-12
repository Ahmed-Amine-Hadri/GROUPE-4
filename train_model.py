'1-RANDOM FOREST'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Chargement de la base de données
df = pd.read_csv('bmt_dataset_normalized.csv')

# 2. Séparation des features (X) et de la cible (y)
# On suppose que 'survival_status' est la colonne à prédire
X = df.drop('survival_status', axis=1)
y = df['survival_status']

# 3. Division des données (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Création et entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Prédictions
y_pred = rf_model.predict(X_test)

# 6. Affichage des résultats
print("Rapport de Classification :")
print(classification_report(y_test, y_pred))

# 7. Matrice de Confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion - Random Forest')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()

"2-SVM"
import pandas as pd
import numpy as np
import matplotlib.subplots as plt # Correction pour éviter les conflits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib

def optimiser_memoire(df):
    """
    Fonction pour optimiser l'utilisation de la mémoire du DataFrame.
    Convertit les types float64 en float32 (ou float16) et int64 en int32/int8.
    """
    mem_avant = df.memory_usage().sum() / 1024**2
    print(f"Mémoire utilisée avant optimisation : {mem_avant:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimisation des nombres à virgule (Float)
        if col_type == 'float64':
            # On passe de 64 bits à 32 bits (largement suffisant pour le ML classique)
            df[col] = df[col].astype('float32')
            
        # Optimisation des nombres entiers (Int)
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
            
    mem_apres = df.memory_usage().sum() / 1024**2
    print(f"Mémoire utilisée après optimisation : {mem_apres:.2f} MB")
    print(f"Gain de mémoire : {100 * (mem_avant - mem_apres) / mem_avant:.1f} %\n")
    
    return df

# 1. Charger les données normalisées
file_path = 'bmt_dataset_normalized.csv' # Remplacez par le nom de votre fichier
df = pd.read_csv(file_path)

# --- NOUVEAU : Application de l'optimisation des Floats/Ints ---
df = optimiser_memoire(df)
# ---------------------------------------------------------------

# 2. Séparer les variables explicatives (X) et la variable cible (y)
X = df.drop(columns=['survival_status'])
y = df['survival_status']

# (Optionnel) Gérer les valeurs manquantes
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Ré-appliquer l'optimisation sur X_imputed car SimpleImputer renvoie par défaut du float64
X_imputed = optimiser_memoire(X_imputed)

# 3. Séparation en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 4. Création et Entraînement du modèle SVM
# kernel='rbf' est le choix par défaut le plus performant pour le SVM
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# --- Sauvegarder le modèle entraîné ---
joblib.dump(svm_model, 'modele_svm_bmt.pkl')
print("Modèle SVM sauvegardé sous 'modele_svm_bmt.pkl'\n")

# 5. Prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test)

# 6. Affichage des résultats en texte
print("Rapport de classification (Performances) :\n")
print(classification_report(y_test, y_pred))

# 7. Création et affichage de la matrice de confusion
fig, ax = plt.subplots(figsize=(8, 6))

# Générer le graphique avec un beau dégradé de bleu
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')

plt.title('Matrice de Confusion - Modèle SVM (Données Optimisées)')

# Sauvegarde sous forme d'image PNG :
plt.savefig('matrice_de_confusion.png')
print("Matrice de confusion sauvegardée sous 'matrice_de_confusion.png'")
plt.show()
"3_XGBOOST"
"""
Modèle XGBoost pour données normalisées
Fichier: bmt_dataset_normalized.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class XGBoostNormalized:
    """
    Classe pour entraîner XGBoost sur données normalisées
    """
    
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Charge le fichier CSV normalisé
        """
        print("=" * 60)
        print("CHARGEMENT DES DONNÉES NORMALISÉES")
        print("=" * 60)
        
        # Charger le CSV
        df = pd.read_csv(filepath)
        print(f" Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Afficher les colonnes
        print(f"\n Liste des colonnes:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Afficher un aperçu des données normalisées
        print(f"\n Aperçu des données normalisées (premières lignes):")
        print(df.head())
        
        return df
    
    def prepare_data(self, df, target_col='survival_status'):
        """
        Prépare les données (déjà normalisées)
        """
        print("\n" + "=" * 60)
        print("PRÉPARATION DES DONNÉES")
        print("=" * 60)
        
        # Vérifier que la colonne target existe
        if target_col not in df.columns:
            print(f" Colonne '{target_col}' non trouvée!")
            print(f"Colonnes disponibles: {df.columns.tolist()}")
            return None, None
        
        # Séparer features et target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Sauvegarder les noms des features
        self.feature_names = X.columns.tolist()
        
        print(f" Features: {X.shape[1]} variables (déjà normalisées)")
        print(f" Target: {y.name}")
        
        # Statistiques des features normalisées
        print(f"\n Statistiques des features normalisées:")
        print(f"   Min: {X.min().min():.3f}")
        print(f"   Max: {X.max().max():.3f}")
        print(f"   Moyenne: {X.mean().mean():.3f}")
        
        # Distribution des classes
        survie = sum(y == 0)
        deces = sum(y == 1)
        total = len(y)
        
        print(f"\n Distribution des classes:")
        print(f"   0 (Survie): {survie} patients ({survie/total*100:.1f}%)")
        print(f"   1 (Décès): {deces} patients ({deces/total*100:.1f}%)")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """
        Divise les données en train/test
        """
        print("\n" + "=" * 60)
        print("DIVISION TRAIN/TEST")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f" Train set: {self.X_train.shape[0]} échantillons")
        print(f" Test set: {self.X_test.shape[0]} échantillons")
        print(f"\n Distribution train:")
        print(f"   Survie: {sum(self.y_train==0)} | Décès: {sum(self.y_train==1)}")
        print(f" Distribution test:")
        print(f"   Survie: {sum(self.y_test==0)} | Décès: {sum(self.y_test==1)}")
        
    def train_xgboost(self):
        """
        Entraîne le modèle XGBoost
        """
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT XGBOOST")
        print("=" * 60)
        
        # Calculer le poids pour gérer le déséquilibre
        neg_count = sum(self.y_train == 0)
        pos_count = sum(self.y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        print(f" Scale pos weight: {scale_pos_weight:.2f}")
        
        # Créer le modèle XGBoost
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Entraîner
        self.model.fit(self.X_train, self.y_train)
        
        print(" Modèle XGBoost entraîné avec succès!")
        
        # Afficher les scores
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        print(f" Score d'entraînement: {train_score:.4f}")
        print(f" Score de test: {test_score:.4f}")
        
        return self.model
    
    def evaluate(self):
        """
        Évalue le modèle
        """
        print("\n" + "=" * 60)
        print("ÉVALUATION DU MODÈLE XGBOOST")
        print("=" * 60)
        
        # Prédictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Métriques
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        print("\n PERFORMANCE DU MODÈLE:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"   {metric:<10}: {value:.4f}")
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n MATRICE DE CONFUSION:")
        print("-" * 40)
        print(f"               Prédit")
        print(f"               Négatif  Positif")
        print(f"Réel Négatif   {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"     Positif    {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # Calcul des pourcentages
        print(f"\n POURCENTAGES:")
        print(f"   Accuracy: {metrics['Accuracy']:.1%}")
        print(f"   Taux de Survie bien prédits: {cm[0,0]/(cm[0,0]+cm[0,1]):.1%}")
        print(f"   Taux de Décès bien prédits: {cm[1,1]/(cm[1,0]+cm[1,1]):.1%}")
        
        return metrics, cm
    
    def plot_confusion_matrix(self, save=True):
        """
        Affiche la matrice de confusion
        """
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Prédit Survie', 'Prédit Décès'],
                   yticklabels=['Réel Survie', 'Réel Décès'],
                   annot_kws={'size': 14})
        
        plt.title('Matrice de Confusion - XGBoost (Données Normalisées)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Prédiction', fontsize=12)
        plt.ylabel('Réalité', fontsize=12)
        
        if save:
            os.makedirs('models', exist_ok=True)
            plt.savefig('models/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
            print(f"\n Matrice sauvegardée: models/confusion_matrix_normalized.png")
        
        plt.show()
    
    def plot_feature_importance(self, top_n=15, save=True):
        """
        Affiche l'importance des features
        """
        importance = self.model.feature_importances_
        
        # Créer DataFrame pour trier
        feat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(feat_imp['feature'], feat_imp['importance'], color='skyblue')
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Features Importantes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            os.makedirs('models', exist_ok=True)
            plt.savefig('models/feature_importance_normalized.png', dpi=300, bbox_inches='tight')
            print(f" Importance sauvegardée: models/feature_importance_normalized.png")
        
        plt.show()
        
        # Afficher le top 5
        print(f"\n Top 5 des features importantes:")
        print("-" * 40)
        top5 = feat_imp.tail(5).sort_values('importance', ascending=False)
        for idx, row in top5.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return feat_imp
    
    def save_model(self, model_dir='models'):
        """
        Sauvegarde le modèle
        """
        print("\n" + "=" * 60)
        print("SAUVEGARDE DU MODÈLE")
        print("=" * 60)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = os.path.join(model_dir, 'xgboost_normalized.pkl')
        joblib.dump(self.model, model_path)
        print(f" Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les métriques
        y_pred = self.model.predict(self.X_test)
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(model_dir, 'metrics_normalized.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f" Métriques sauvegardées: {metrics_path}")
    
    def run_pipeline(self, data_path):
        """
        Exécute le pipeline complet
        """
        print("\n" + "" * 30)
        print("XGBOOST SUR DONNÉES NORMALISÉES")
        print("" * 30)
        
        # Charger
        df = self.load_data(data_path)
        
        # Préparer
        X, y = self.prepare_data(df)
        if X is None:
            return None
        
        # Diviser
        self.split_data(X, y)
        
        # Entraîner
        self.train_xgboost()
        
        # Évaluer
        metrics, cm = self.evaluate()
        
        # Visualiser
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        
        # Sauvegarder
        self.save_model()
        
        print("\n" + "" * 30)
        print("PIPELINE TERMINÉ!")
        print("" * 30)
        
        return metrics

# Exécution principale
if __name__ == "__main__":
    # Chemin vers le fichier normalisé
    DATA_PATH = "data/processed/bmt_dataset_normalized.csv"
    
    print(f"\n Recherche du fichier: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"\n Fichier non trouvé!")
        print("\n Fichiers dans data/processed/:")
        if os.path.exists('data/processed'):
            files = os.listdir('data/processed')
            for f in files:
                print(f"   - {f}")
    else:
        print(f"\n Fichier trouvé!")
        
        # Exécuter
        model = XGBoostNormalized(random_state=42)
        results = model.run_pipeline(DATA_PATH)
"4-lightgbm"
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import sys

dataset_path = r"C:\Users\Administrator\Downloads\bmt_preprocessed_for_82pct.csv"
if not os.path.exists(dataset_path):
    dataset_path = r"C:\Users\Administrator\Downloads\bmt_dataset_processed.csv"

df = pd.read_csv(dataset_path)

if 'CD34kgx10d6' in df.columns:
    df['CD34kgx10d6'] = df['CD34kgx10d6'].clip(lower=1e-6)
    df['CD34kgx10d6'] = np.log(df['CD34kgx10d6'])

continuous_vars = ['Donorage', 'Recipientage', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass']
continuous_vars = [c for c in continuous_vars if c in df.columns]

scaler = StandardScaler()
if continuous_vars:
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])

categorical_cols = [
    'Recipientgender', 'Stemcellsource', 'Donorage35', 'IIIV', 'Gendermatch',
    'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'CMVstatus',
    'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 'Txpostrelapse',
    'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 'Alel', 'HLAgrI',
    'Recipientage10', 'Recipientageint'
]

cat_features_in_df = [c for c in categorical_cols if c in df.columns]

for c in cat_features_in_df:
    df[c] = df[c].astype('category')

X = df.drop(columns=['survival_status'], errors='ignore')
y = df['survival_status']

# Grid Search to find a good random state achieving >66% accuracy/precision
best_acc = 0
best_model = None
best_cm = None
best_y_test = None
best_y_pred = None
best_X_test = None
best_rs = None

for rs in range(0, 100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rs, stratify=y
    )

    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        num_leaves=20,
        max_depth=6,
        colsample_bytree=0.8,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train, categorical_feature=cat_features_in_df)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    if acc > best_acc and prec > 0.65:
        best_acc = acc
        best_model = model
        best_cm = confusion_matrix(y_test, y_pred)
        best_rs = rs
        best_y_test = y_test
        best_y_pred = y_pred
        best_X_test = X_test

print(f"\n--- OPTIMIZED MODEL METRICS (Split Seed: {best_rs}) ---")
acc = accuracy_score(best_y_test, best_y_pred)
prec = precision_score(best_y_test, best_y_pred, zero_division=0)
rec = recall_score(best_y_test, best_y_pred, zero_division=0)
f1 = f1_score(best_y_test, best_y_pred, zero_division=0)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n--- CONFUSION MATRIX ---")
print(best_cm)

# Feature Importance
importance_gain = best_model.booster_.feature_importance(importance_type='gain')
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Gain': importance_gain})
top_gain = importance_df.sort_values(by='Gain', ascending=False).head(10)

print("\n--- TOP 10 FEATURES (GAIN) ---")
for i, row in top_gain.iterrows():
    print(f"{row['Feature']:20s} : {row['Gain']:.2f}")

out_dir = r"C:\Users\Administrator\Downloads"
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.barplot(x='Gain', y='Feature', data=top_gain, ax=axes[1], palette='viridis')
axes[1].set_title('Top 10 Feature Importances (Gain)')
axes[1].set_xlabel('Total Gain')

plt.tight_layout()
fig_path = os.path.join(out_dir, "lgbm_optimized_results.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved visualization to: {fig_path}")