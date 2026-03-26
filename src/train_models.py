import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def save_confusion_matrix(model, X_test, y_test, output_path="outputs/confusion_matrix.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Risk", "High Risk"]
    )
    disp.plot(cmap="Blues")

    plt.title("Random Forest Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def train_models(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    metrics = pd.DataFrame([
        evaluate_model("Logistic Regression", lr, X_test, y_test),
        evaluate_model("Random Forest", rf, X_test, y_test),
    ])

    return lr, rf, metrics


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def train_models(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    metrics = pd.DataFrame([
        evaluate_model("Logistic Regression", lr, X_test, y_test),
        evaluate_model("Random Forest", rf, X_test, y_test),
    ])

    # Save confusion matrix for Random Forest
    save_confusion_matrix(rf, X_test, y_test)	

    return lr, rf, metrics
