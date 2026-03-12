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
    DATA_PATH = "data/processed/bmt_preprocessed_for_82pct.csv"
    
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