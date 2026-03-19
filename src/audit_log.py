import os
import pandas as pd
from datetime import datetime


def create_audit_log(model, X_test, output_dir="outputs", model_version="rf_v1"):
    os.makedirs(output_dir, exist_ok=True)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    audit_df = pd.DataFrame({
        "timestamp": [datetime.utcnow().isoformat()] * len(X_test),
        "case_id": list(range(1, len(X_test) + 1)),
        "model_version": [model_version] * len(X_test),
        "prediction": predictions,
        "probability_default": probabilities,
    })

    audit_df.to_csv(f"{output_dir}/audit_log.csv", index=False)
    return audit_df
