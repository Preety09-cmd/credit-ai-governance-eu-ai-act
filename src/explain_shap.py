import os
import shap
import matplotlib.pyplot as plt
import numpy as np


def run_shap(model, X_train, X_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary-classification output shape safely
    if isinstance(shap_values, list):
        shap_for_plot = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        shap_for_plot = shap_values[:, :, 1]
    else:
        shap_for_plot = shap_values

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_for_plot, X_test, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # Local explanation for one sample
    sample_index = 0

    expected_value = explainer.expected_value
    if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
        if np.array(expected_value).ndim > 0:
            expected_value = expected_value[1]
        else:
            expected_value = float(expected_value)

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_for_plot[sample_index],
            base_values=expected_value,
            data=X_test.iloc[sample_index],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    plt.savefig(f"{output_dir}/shap_local_explanation.png", bbox_inches="tight")
    plt.close()

    return shap_values
