import os


def create_governance_report(metrics_df, missing_report, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    report_path = f"{output_dir}/governance_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("AI Governance Report\n")
        f.write("===================\n\n")

        f.write("1. Data Quality Review\n")
        f.write("----------------------\n")
        f.write(str(missing_report))
        f.write("\n\n")

        f.write("2. Model Performance Comparison\n")
        f.write("-------------------------------\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("3. Governance Notes\n")
        f.write("-------------------\n")
        f.write("- Logistic Regression used as interpretable baseline model.\n")
        f.write("- Random Forest used as higher-complexity predictive model.\n")
        f.write("- SHAP and LIME applied for model explainability.\n")
        f.write("- Audit logging enabled for prediction traceability.\n")
        f.write("- Monitoring logic included for unusual prediction patterns.\n")
