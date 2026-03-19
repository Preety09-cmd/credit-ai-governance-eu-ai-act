import pandas as pd


def run_monitoring(audit_log_path="outputs/audit_log.csv"):
    df = pd.read_csv(audit_log_path)

    alerts = []

    high_risk_ratio = (df["prediction"] == 1).mean()
    if high_risk_ratio > 0.70:
        alerts.append("Alert: unusually high proportion of high-risk predictions detected.")

    if df["probability_default"].max() > 0.98:
        alerts.append("Alert: extremely high prediction confidence detected.")

    if not alerts:
        alerts.append("No major anomalies detected.")

    with open("outputs/monitoring_alerts.txt", "w", encoding="utf-8") as f:
        for alert in alerts:
            f.write(alert + "\n")

    return alerts
