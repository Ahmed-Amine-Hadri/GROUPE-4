# GROUPE-4
*data processing*
analyze_distributions : Identifie les variables numériques trop asymétriques (skewness > 0.75) afin de cibler celles qui nécessitent une transformation pour stabiliser la variance.

analyze_correlations : Repère les variables redondantes via une matrice de Spearman ; on l'utilise pour éviter la multicolinéarité qui fausse l'importance des caractéristiques.

optimize_memory : Convertit les types de données (ex: float64 en float32) pour réduire l'empreinte RAM et accélérer les calculs sans perte de précision significative.

clean_data : Supprime les colonnes inutiles (ID, dates), impute les valeurs manquantes et encode les catégories en variables muettes pour rendre le dataset exploitable par un modèle.

handle_outliers : Écrête (clipping) les valeurs extrêmes aux percentiles 1% et 99% pour empêcher les données aberrantes de biaiser l'apprentissage.

#apply_log_transformations : Applique le logarithme sur les colonnes asymétriques identifiées plus haut pour normaliser leur distribution et améliorer la performance prédictive.

 reduce_multicollinearity : Supprime les variables trop corrélées entre elles (R > 0.80) pour simplifier le modèle et limiter le risque de surapprentissage (overfitting).
*train models*
# XGBoost (Extreme Gradient Boosting)
C’est un algorithme de boosting qui construit des arbres de décision de manière séquentielle. Chaque nouvel arbre tente de corriger les erreurs de prédiction des arbres précédents.

Pondération des classes (scale_pos_weight) : Le code calcule dynamiquement le ratio entre les patients décédés et les survivants. C'est crucial pour forcer l'IA à ne pas ignorer la classe minoritaire (souvent les décès dans ce type de dataset).

Paramètres de contrôle : max_depth=6 limite la profondeur des arbres pour éviter que le modèle n'apprenne par cœur des détails inutiles (overfitting). learning_rate=0.1 assure une progression lente mais stable de l'apprentissage.

Analyse d'importance : Le code génère un graphique des "Top 15 Features". Cela permet de voir techniquement quelles variables biologiques (comme le dosage CD34) ont le plus pesé dans la décision finale du modèle.
# SVM (Support Vector Machine)
Le SVM cherche à tracer une frontière (un hyperplan) qui sépare le plus largement possible les deux groupes (Survie vs Décès).

Imputation (SimpleImputer) : Contrairement aux arbres, le SVM est mathématiquement incapable de gérer les valeurs manquantes (NaN). Le code utilise donc la médiane pour remplir les trous avant de présenter les données au modèle.

Noyau RBF (kernel='rbf') : Ce noyau permet de créer une frontière de décision non linéaire (courbe). C'est indispensable car, en biologie, les relations entre les variables sont rarement de simples lignes droites.

Probabilités : probability=True est activé pour que le modèle ne donne pas juste un "oui/non", mais un score de confiance (ex: 85% de chances de survie).
# Random Forest
C’est une méthode de "Bagging". On crée 100 arbres de décision différents qui votent. La décision finale est celle de la majorité.

Stabilité : Avec n_estimators=100, le code construit une forêt robuste. Si un arbre fait une erreur isolée, elle est compensée par les 99 autres.

Reproductibilité : Le random_state=42 est fixé pour que l'aspect aléatoire de la forêt soit identique à chaque exécution, facilitant le débogage.

Rapport de performance : L'utilisation de classification_report permet d'analyser le score F1, qui est l'équilibre parfait entre la précision (ne pas se tromper de diagnostic) et le rappel (détecter tous les patients à risque).
# LightGBM (Light Gradient Boosting Machine)
Une variante ultra-rapide du boosting qui utilise une croissance des arbres par "feuilles" plutôt que par "niveaux"

Prétraitement manuel : Le code applique un StandardScaler et un passage au Logarithme. Cela normalise les échelles de données (ex: l'âge entre 0 et 80 vs les globules blancs en milliers) pour que le modèle ne soit pas perturbé par les grands nombres.

Optimisation par boucle (range(0, 100)) : C'est la partie la plus avancée. Le code teste 100 graines aléatoires différentes pour le découpage Train/Test. Il cherche la "meilleure graine" qui maximise l'Accuracy tout en gardant une Precision décente (> 0.65).

Importance par Gain : Le code analyse le Gain (la réduction totale de l'incertitude apportée par chaque variable), ce qui offre une vision plus précise de l'utilité réelle de chaque caractéristique biologique que le simple comptage de fréquence.

*evaluation*
#
#
#
#
##
#
#
#
#
#
#
#
#
#
#
#
#