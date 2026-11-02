import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load independently verified test assignments.
df = pd.read_csv("data/assignment_evaluation.csv")

required_columns = {"actual_team", "predicted_team"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.dropna(subset=["actual_team", "predicted_team"])

if df.empty:
    raise ValueError("No valid evaluation records were found.")

y_true = df["actual_team"].astype(str)
y_pred = df["predicted_team"].astype(str)

print("\n=== ASSIGNMENT MODEL EVALUATION ===")
print(f"Evaluated work orders: {len(df)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\n=== PRECISION, RECALL, F1-SCORE, SUPPORT ===")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)

labels = sorted(set(y_true) | set(y_pred))

print("\n=== CONFUSION MATRIX ===")
print("Team labels:", labels)
print(confusion_matrix(y_true, y_pred, labels=labels))