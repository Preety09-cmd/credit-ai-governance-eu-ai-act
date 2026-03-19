import os
from src.preprocess import load_and_preprocess
from src.train_models import train_models
from src.explain_shap import run_shap
from src.explain_lime import run_lime
from src.governance_checks import create_governance_report
from src.audit_log import create_audit_log
from src.monitor import run_monitoring


def main():
    os.makedirs("outputs", exist_ok=True)

    dataset_path = "data/german_credit.csv"
    target_column = "target"

    print("Loading and preprocessing dataset...")
    df, X_train, X_test, y_train, y_test, missing_report, encoders = load_and_preprocess(
        dataset_path, target_col=target_column
    )

    print("Training models...")
    lr_model, rf_model, metrics_df = train_models(X_train, y_train, X_test, y_test)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)

    print("Generating SHAP explanations...")
    run_shap(rf_model, X_train, X_test, output_dir="outputs")

    print("Generating LIME explanation...")
    run_lime(rf_model, X_train, X_test, output_dir="outputs")

    print("Creating governance report...")
    create_governance_report(metrics_df, missing_report, output_dir="outputs")

    print("Creating audit log...")
    create_audit_log(rf_model, X_test, output_dir="outputs", model_version="rf_v1")

    print("Running monitoring checks...")
    alerts = run_monitoring("outputs/audit_log.csv")

    print("\nWorkflow completed successfully.")
    print("Monitoring alerts:")
    for alert in alerts:
        print("-", alert)


if __name__ == "__main__":
    main()
