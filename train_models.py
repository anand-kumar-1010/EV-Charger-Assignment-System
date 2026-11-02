"""
Train and compare EV charger installation-duration regression models.

Model selection uses cross-validation on the training split.
The test split is held out until the final evaluation.

IMPORTANT:
The current historical dataset is synthetic demonstration data.
Metrics do not establish real-world performance.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "historical_installations_synthetic.csv"
MODELS = ROOT / "models"

TARGET = "actual_duration_hours"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

MODELS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

if not DATA.exists():
    raise FileNotFoundError(f"Historical dataset not found: {DATA}")

df = pd.read_csv(DATA)

if df.empty:
    raise ValueError("The historical dataset is empty.")

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' is missing.")

# Convert target to numeric and discard rows without a valid target.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# Installation duration must be positive.
df = df[df[TARGET] > 0].copy()

if len(df) < CV_FOLDS * 2:
    raise ValueError(
        f"At least {CV_FOLDS * 2} valid records are required "
        f"for {CV_FOLDS}-fold cross-validation."
    )

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].astype(float)

if X.shape[1] == 0:
    raise ValueError("No input features were found.")

# Keep complexity categorical even if values happen to be numeric.
if "complexity" in X.columns:
    X["complexity"] = X["complexity"].astype("string")

print("=" * 70)
print("EV CHARGER INSTALLATION DURATION MODEL TRAINING")
print("=" * 70)
print(f"Dataset: {DATA.name}")
print(f"Valid records: {len(df)}")
print(f"Features: {list(X.columns)}")
print(f"Target: {TARGET}")
print(f"scikit-learn version: {sklearn.__version__}")


# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

numeric_features = [
    col for col in X.columns if col not in categorical_features
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

transformers = []

if numeric_features:
    transformers.append(("numeric", numeric_pipeline, numeric_features))

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


# --------------------------------------------------
# 4. SPLIT ONCE: TEST DATA IS HELD OUT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nData split:")
print(f"Training records: {len(X_train)}")
print(f"Untouched test records: {len(X_test)}")


# --------------------------------------------------
# 5. DEFINE CANDIDATE MODELS
# --------------------------------------------------

candidates = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    ),

    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        loss="huber",
        random_state=RANDOM_STATE,
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# --------------------------------------------------
# 6. CROSS-VALIDATION MODEL SELECTION
# --------------------------------------------------

cv = KFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

cv_results = {}
pipelines = {}

print("\n" + "=" * 70)
print(f"{CV_FOLDS}-FOLD CROSS-VALIDATION (TRAINING DATA ONLY)")
print("=" * 70)

for name, estimator in candidates.items():
    pipeline = Pipeline(
        steps=[
            ("prepare", preprocessor),
            ("model", estimator),
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )

    mean_mae = -scores["test_mae"].mean()
    std_mae = scores["test_mae"].std()
    mean_rmse = -scores["test_rmse"].mean()
    std_rmse = scores["test_rmse"].std()
    mean_r2 = scores["test_r2"].mean()
    std_r2 = scores["test_r2"].std()

    cv_results[name] = {
        "CV_MAE_mean_hours": round(float(mean_mae), 4),
        "CV_MAE_std_hours": round(float(std_mae), 4),
        "CV_RMSE_mean_hours": round(float(mean_rmse), 4),
        "CV_RMSE_std_hours": round(float(std_rmse), 4),
        "CV_R2_mean": round(float(mean_r2), 4),
        "CV_R2_std": round(float(std_r2), 4),
    }

    pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"  CV MAE:  {mean_mae:.4f} ± {std_mae:.4f} hours")
    print(f"  CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} hours")
    print(f"  CV R²:   {mean_r2:.4f} ± {std_r2:.4f}")


# Select using cross-validation RMSE only, not test-set performance.
selected_model = min(
    cv_results,
    key=lambda name: cv_results[name]["CV_RMSE_mean_hours"],
)

best_pipeline = pipelines[selected_model]

print("\nSelected using lowest mean cross-validation RMSE:")
print(selected_model)


# --------------------------------------------------
# 7. FIT SELECTED MODEL AND EVALUATE ON TEST SET
# --------------------------------------------------

best_pipeline.fit(X_train, y_train)
test_predictions = best_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

test_metrics = {
    "MAE_hours": round(float(test_mae), 4),
    "RMSE_hours": round(float(test_rmse), 4),
    "R2": round(float(test_r2), 4),
}

print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST PERFORMANCE")
print("=" * 70)
print(f"MAE:  {test_mae:.4f} hours")
print(f"RMSE: {test_rmse:.4f} hours")
print(f"R²:   {test_r2:.4f}")


# --------------------------------------------------
# 8. PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

# This measures how much test score changes when one raw input column
# is shuffled. It is a diagnostic, not proof of causation.
feature_importance = {}

try:
    importance = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    sorted_indices = np.argsort(importance.importances_mean)[::-1]

    for index in sorted_indices:
        feature_importance[str(X_test.columns[index])] = {
            "mean": round(float(importance.importances_mean[index]), 6),
            "std": round(float(importance.importances_std[index]), 6),
        }

    print("\nPermutation feature importance (higher mean = more useful):")
    for feature, values in feature_importance.items():
        print(f"  {feature}: {values['mean']:.6f}")

except Exception as exc:
    warnings.warn(f"Could not calculate permutation importance: {exc}")


# --------------------------------------------------
# 9. SAVE MODEL AND METRICS
# --------------------------------------------------

model_path = MODELS / "duration_model.joblib"
metrics_path = MODELS / "model_metrics.json"
comparison_path = MODELS / "model_comparison.csv"

# Keep the filename unchanged so assignment_engine.py can load it.
joblib.dump(best_pipeline, model_path)

comparison = pd.DataFrame(cv_results).T
comparison = comparison.sort_values("CV_RMSE_mean_hours")
comparison.to_csv(comparison_path, index=True, index_label="Model")

metrics = {
    "selected_model": selected_model,
    "selection_criterion": f"Lowest mean {CV_FOLDS}-fold CV RMSE on training data",
    "dataset": DATA.name,
    "total_records": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "cv_folds": CV_FOLDS,
    "cross_validation_results": cv_results,
    "test_results": test_metrics,
    "feature_columns": list(X.columns),
    "permutation_feature_importance": feature_importance,
    "sklearn_version": sklearn.__version__,
    "data_note": (
        "The historical dataset is synthetic demonstration data. "
        "These metrics do not establish real-world performance."
    ),
}

metrics_path.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("MODEL COMPARISON — CROSS-VALIDATION")
print("=" * 70)
print(comparison.to_string())

print("\nSaved files:")
print(f"  Model:      {model_path}")
print(f"  Metrics:    {metrics_path}")
print(f"  Comparison: {comparison_path}")

print("\nIMPORTANT:")
print("- Model selection used cross-validation on training data.")
print("- The test set was used only for final evaluation.")
print("- Results are based on synthetic data; validate with real records.")
print("- Keep scikit-learn versions consistent when loading the joblib model.")
