import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(path: str, target_col: str = "target"):
    df = pd.read_csv(path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Missing value report
    missing_report = df.isnull().sum()

    # Simple fill for missing values if any
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return df, X_train, X_test, y_train, y_test, missing_report, encoders
