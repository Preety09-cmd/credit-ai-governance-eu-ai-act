import os
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def run_lime(model, X_train, X_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=["Low Risk", "High Risk"],
        mode="classification"
    )

    sample_index = 0

    explanation = explainer.explain_instance(
        data_row=np.array(X_test.iloc[sample_index]),
        predict_fn=model.predict_proba
    )

    explanation.save_to_file(f"{output_dir}/lime_explanation.html")

    return explanation
