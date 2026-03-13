import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Trainer class for XGBoost model.
    """

    def __init__(self, project_root):
        """
        Initialize paths and logger.
        """
        self.project_root = Path(project_root)
        self.data_path = self.project_root / "data" / "processed" / "final_dataset.csv"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.logger = logging.getLogger(__name__)

    def train_xgboost(self):
        """
        Train an XGBoost classifier and save outputs.
        """
        self.logger.info("Running XGBoost model training...")

        print("Dataset path:", self.data_path)
        print("Exists:", self.data_path.exists())

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        # Load dataset
        df = pd.read_csv(self.data_path)

        # Target and features
        target_col = "survival_status"

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_names = X.columns.tolist()

        # Imputation for missing values
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Class imbalance handling
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        # Model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
        }

        print("\n===== XGBOOST METRICS =====")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            annot_kws={"size": 14}
        )
        plt.title("Confusion Matrix - XGBoost")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(self.models_dir / "xgboost_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # Feature importance
        importance = model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=True).tail(15)

        plt.figure(figsize=(12, 8))
        plt.barh(feat_imp["feature"], feat_imp["importance"])
        plt.title("Top 15 Important Features - XGBoost")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(self.models_dir / "xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        # Save model and imputer
        joblib.dump(model, self.models_dir / "xgboost_model.pkl")
        joblib.dump(imputer, self.models_dir / "xgboost_imputer.pkl")

        print("\nFiles saved in:", self.models_dir)
        print("- xgboost_model.pkl")
        print("- xgboost_imputer.pkl")
        print("- xgboost_confusion_matrix.png")
        print("- xgboost_feature_importance.png")


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent
    trainer = ModelTrainer(project_root=ROOT_DIR)

    print("XGBoost training started")
    trainer.train_xgboost()