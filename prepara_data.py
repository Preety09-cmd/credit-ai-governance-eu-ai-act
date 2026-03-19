import pandas as pd

# Load raw dataset
df = pd.read_csv("data/german.data-numeric", sep=r"\s+", header=None, engine="python")

# Assign generic column names: 24 features + 1 target
df.columns = [f"feature_{i}" for i in range(1, 25)] + ["target"]

# Convert target: 1=good, 2=bad  →  0=low risk, 1=high risk
df["target"] = df["target"].apply(lambda x: 1 if x == 2 else 0)

# Save cleaned CSV
df.to_csv("data/german_credit.csv", index=False)

print("✅ Dataset prepared successfully!")
print(df.head())
print("Shape:", df.shape)
