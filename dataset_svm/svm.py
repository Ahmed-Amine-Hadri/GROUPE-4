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
file_path = 'bmt_preprocessed_for_82pct.csv' # Remplacez par le nom de votre fichier
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