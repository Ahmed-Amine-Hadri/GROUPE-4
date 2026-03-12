
This branch focuses on statistical optimization and dimensionality reduction to improve model performance.Key Contributions:
Log Transformations (log.py): Applied to right-skewed features like CD34kgx10d6. This normalizes the distribution so the model isn't biased by extreme values.
Multicollinearity Reduction (Heat-map.py): Used Spearman correlation to find redundant features. I kept only the strongest predictor from any pair correlating $> 0.80$, reducing noise.
Outlier Clipping (OutLayers.py): Used 1st and 99th percentile clipping (Winsorization) to handle clinical anomalies without losing data rows.
EDA (eda.ipynb & Data--analysis.py): Verified that transformations actually improved data distributions before integration.
