def generate_xgboost_shap_analysis(self):
    """
    Generate SHAP explanations, plots, and automatic interpretation
    for the trained XGBoost model.

    Target mapping:
        1 = survived
        0 = not survived
    """

    self.logger.info("Generating SHAP analysis for XGBoost...")

    import joblib
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # -----------------------------
    # 1) Load dataset
    # -----------------------------
    df = pd.read_csv(self.data_path)

    target_col = "survival_status"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Recreate same split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 2) Load trained model
    # -----------------------------
    model_path = self.models_dir / "xgboost_model.pkl"
    model = joblib.load(model_path)

    # -----------------------------
    # 3) Build SHAP explainer
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classification compatibility
    if isinstance(shap_values, list):
        shap_values_to_use = shap_values[1]   # class 1 = survived
    else:
        shap_values_to_use = shap_values

    # Handle expected value
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        if len(np.atleast_1d(expected_value)) > 1:
            expected_value_to_use = np.atleast_1d(expected_value)[1]
        else:
            expected_value_to_use = np.atleast_1d(expected_value)[0]
    else:
        expected_value_to_use = expected_value

    # -----------------------------
    # 4) Global feature analysis
    # -----------------------------
    mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
    mean_signed_shap = shap_values_to_use.mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance_df.to_csv(
        self.models_dir / "xgboost_top_features_analysis.csv",
        index=False
    )

    # -----------------------------
    # 5) SHAP summary plot
    # -----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_test,
        show=False
    )
    plt.title("SHAP Summary Plot - XGBoost")
    plt.tight_layout()
    plt.savefig(
        self.models_dir / "xgboost_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # 6) SHAP bar plot
    # -----------------------------
    plt.figure()
    shap.summary_plot(
        shap_values_to_use,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Bar) - XGBoost")
    plt.tight_layout()
    plt.savefig(
        self.models_dir / "xgboost_shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # 7) Dependence plots for top 3 features
    # -----------------------------
    top_features = shap_importance_df["feature"].head(3).tolist()

    for feature_name in top_features:
        plt.figure()
        shap.dependence_plot(
            feature_name,
            shap_values_to_use,
            X_test,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature_name}")
        plt.tight_layout()
        safe_name = str(feature_name).replace("/", "_").replace(" ", "_")
        plt.savefig(
            self.models_dir / f"xgboost_dependence_{safe_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # -----------------------------
    # 8) Waterfall plot for one sample
    # -----------------------------
    sample_idx = 0

    explanation = shap.Explanation(
        values=shap_values_to_use[sample_idx],
        base_values=expected_value_to_use,
        data=X_test.iloc[sample_idx].values,
        feature_names=X_test.columns.tolist()
    )

    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(
        self.models_dir / "xgboost_shap_waterfall_sample0.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -----------------------------
    # 9) Prediction info for one sample
    # -----------------------------
    pred_class = int(model.predict(X_test.iloc[[sample_idx]])[0])
    pred_proba = model.predict_proba(X_test.iloc[[sample_idx]])[0]
    true_label = int(y_test.iloc[sample_idx])

    # -----------------------------
    # 10) Automatic text interpretation
    # -----------------------------
    top10 = shap_importance_df.head(10)

    interpretation_lines = []
    interpretation_lines.append("XGBoost SHAP Interpretation Report")
    interpretation_lines.append("=" * 50)
    interpretation_lines.append("")
    interpretation_lines.append("Target mapping:")
    interpretation_lines.append("1 = survived")
    interpretation_lines.append("0 = not survived")
    interpretation_lines.append("")
    interpretation_lines.append("Interpretation rule:")
    interpretation_lines.append("Positive SHAP value -> pushes prediction toward SURVIVED (class 1)")
    interpretation_lines.append("Negative SHAP value -> pushes prediction toward NOT SURVIVED (class 0)")
    interpretation_lines.append("")
    interpretation_lines.append("Top 10 most influential features based on mean absolute SHAP value:")
    interpretation_lines.append("")

    for i, row in enumerate(top10.itertuples(index=False), start=1):
        direction = (
            "globally pushes more toward SURVIVED"
            if row.mean_signed_shap > 0
            else "globally pushes more toward NOT SURVIVED"
        )
        interpretation_lines.append(
            f"{i}. {row.feature}: mean_abs_shap={row.mean_abs_shap:.6f}, "
            f"mean_signed_shap={row.mean_signed_shap:.6f} -> {direction}"
        )

    interpretation_lines.append("")
    interpretation_lines.append("Sample-level explanation (sample_idx = 0):")
    interpretation_lines.append(f"True label: {true_label}")
    interpretation_lines.append(f"Predicted label: {pred_class}")
    interpretation_lines.append(f"Probability of not survived (class 0): {pred_proba[0]:.4f}")
    interpretation_lines.append(f"Probability of survived (class 1): {pred_proba[1]:.4f}")
    interpretation_lines.append("")

    sample_contrib = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": X_test.iloc[sample_idx].values,
        "shap_value": shap_values_to_use[sample_idx]
    })

    top_positive = sample_contrib.sort_values("shap_value", ascending=False).head(5)
    top_negative = sample_contrib.sort_values("shap_value", ascending=True).head(5)

    interpretation_lines.append("Top 5 features pushing this sample toward SURVIVED:")
    for row in top_positive.itertuples(index=False):
        interpretation_lines.append(
            f"- {row.feature} = {row.feature_value}, shap_value = {row.shap_value:.6f}"
        )

    interpretation_lines.append("")
    interpretation_lines.append("Top 5 features pushing this sample toward NOT SURVIVED:")
    for row in top_negative.itertuples(index=False):
        interpretation_lines.append(
            f"- {row.feature} = {row.feature_value}, shap_value = {row.shap_value:.6f}"
        )

    interpretation_lines.append("")
    interpretation_lines.append("Generated files:")
    interpretation_lines.append("- xgboost_shap_summary.png")
    interpretation_lines.append("- xgboost_shap_bar.png")
    interpretation_lines.append("- xgboost_shap_waterfall_sample0.png")
    interpretation_lines.append("- xgboost_top_features_analysis.csv")
    interpretation_lines.append("- xgboost_shap_interpretation.txt")
    interpretation_lines.append("- xgboost_dependence_<feature>.png for top 3 features")

    with open(self.models_dir / "xgboost_shap_interpretation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(interpretation_lines))

    self.logger.info("SHAP analysis completed successfully for XGBoost.")