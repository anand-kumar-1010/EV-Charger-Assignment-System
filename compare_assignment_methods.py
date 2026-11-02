
from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

ORIGINAL_FILE = OUTPUT_DIR / "assignment_report.csv"
OPTIMIZED_FILE = OUTPUT_DIR / "optimized_assignment_report.csv"
TEAMS_FILE = BASE_DIR / "data" / "teams_with_charger_experience.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD REPORTS
# --------------------------------------------------

original = pd.read_csv(ORIGINAL_FILE)
optimized = pd.read_csv(OPTIMIZED_FILE)
teams = pd.read_csv(TEAMS_FILE)

# Standardize original report columns
original = original.rename(columns={
    "Work Order Number": "work_order_number",
    "Case Number": "case_number",
    "Customer Address": "customer_address",
    "Charger Type": "charger_type",
    "Recommended Team ID": "team_id",
    "Charger-Specific Completed Installations":
        "charger_experience",
    "Distance (km)": "distance_km",
    "Predicted Duration (hours)": "predicted_duration_hours",
    "Assignment Status": "status",
})

# Standardize optimized report columns
optimized = optimized.rename(columns={
    "assigned_team_id": "team_id",
    "charger_specific_installations": "charger_experience",
})

# Ensure work order IDs have consistent types
for df in [original, optimized]:
    df["work_order_number"] = (
        df["work_order_number"].astype(str).str.strip()
    )
    df["team_id"] = df["team_id"].astype(str).str.strip()


# --------------------------------------------------
# 3. VALIDATE INPUT DATA
# --------------------------------------------------

if original["work_order_number"].duplicated().any():
    raise ValueError("Duplicate work orders in original report.")

if optimized["work_order_number"].duplicated().any():
    raise ValueError("Duplicate work orders in optimized report.")

if set(original["work_order_number"]) != set(
    optimized["work_order_number"]
):
    raise ValueError(
        "The two reports do not contain the same work orders."
    )

required_team_columns = {"team_id", "pending_jobs"}

if not required_team_columns.issubset(teams.columns):
    raise ValueError("Team CSV is missing required columns.")

initial_pending = (
    teams.set_index("team_id")["pending_jobs"]
    .astype(int)
    .to_dict()
)


# --------------------------------------------------
# 4. PREPARE COMPARISON
# --------------------------------------------------

comparison = original[
    [
        "work_order_number",
        "case_number",
        "customer_address",
        "charger_type",
        "team_id",
        "charger_experience",
        "distance_km",
        "predicted_duration_hours",
    ]
].rename(columns={
    "team_id": "original_team_id",
    "charger_experience": "original_charger_experience",
    "distance_km": "original_distance_km",
    "predicted_duration_hours": "original_predicted_duration_hours",
})

comparison = comparison.merge(
    optimized[
        [
            "work_order_number",
            "team_id",
            "charger_experience",
            "distance_km",
            "assignment_score",
        ]
    ].rename(columns={
        "team_id": "optimized_team_id",
        "charger_experience": "optimized_charger_experience",
        "distance_km": "optimized_distance_km",
        "assignment_score": "optimized_assignment_score",
    }),
    on="work_order_number",
    how="inner",
    validate="one_to_one",
)

comparison["team_changed"] = (
    comparison["original_team_id"]
    != comparison["optimized_team_id"]
)

comparison["distance_difference_km"] = (
    comparison["optimized_distance_km"]
    - comparison["original_distance_km"]
)

comparison["distance_improvement_km"] = (
    comparison["original_distance_km"]
    - comparison["optimized_distance_km"]
)

comparison["charger_experience_difference"] = (
    comparison["optimized_charger_experience"]
    - comparison["original_charger_experience"]
)


# --------------------------------------------------
# 5. CALCULATE DISTANCE METRICS
# --------------------------------------------------

def distance_metrics(df, column):

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        return {
            "known_distance_count": 0,
            "total_distance_km": None,
            "average_distance_km": None,
            "maximum_distance_km": None,
        }

    return {
        "known_distance_count": int(len(values)),
        "total_distance_km": round(float(values.sum()), 2),
        "average_distance_km": round(float(values.mean()), 2),
        "maximum_distance_km": round(float(values.max()), 2),
    }


original_distance = distance_metrics(
    comparison, "original_distance_km"
)

optimized_distance = distance_metrics(
    comparison, "optimized_distance_km"
)

known_both = comparison.dropna(
    subset=[
        "original_distance_km",
        "optimized_distance_km",
    ]
)

distance_saving = (
    known_both["original_distance_km"].sum()
    - known_both["optimized_distance_km"].sum()
)

distance_improvement_percent = None

if original_distance["total_distance_km"] not in (None, 0):
    distance_improvement_percent = round(
        distance_saving
        / original_distance["total_distance_km"]
        * 100,
        2,
    )


# --------------------------------------------------
# 6. TEAM WORKLOAD COMPARISON
# --------------------------------------------------

team_ids = list(initial_pending.keys())

original_counts = (
    comparison["original_team_id"].value_counts().to_dict()
)

optimized_counts = (
    comparison["optimized_team_id"].value_counts().to_dict()
)

team_rows = []

for team_id in team_ids:

    starting_jobs = initial_pending[team_id]

    original_new = original_counts.get(team_id, 0)
    optimized_new = optimized_counts.get(team_id, 0)

    original_final = starting_jobs + original_new
    optimized_final = starting_jobs + optimized_new

    team_rows.append({
        "team_id": team_id,
        "initial_pending_jobs": starting_jobs,
        "original_new_assignments": original_new,
        "optimized_new_assignments": optimized_new,
        "original_final_pending": original_final,
        "optimized_final_pending": optimized_final,
        "original_capacity_used_percent": round(
            original_final / 5 * 100, 2
        ),
        "optimized_capacity_used_percent": round(
            optimized_final / 5 * 100, 2
        ),
        "original_within_capacity": original_final <= 5,
        "optimized_within_capacity": optimized_final <= 5,
    })

team_comparison = pd.DataFrame(team_rows)


def workload_statistics(column):

    values = team_comparison[column].astype(float)

    return {
        "minimum_final_jobs": int(values.min()),
        "maximum_final_jobs": int(values.max()),
        "average_final_jobs": round(float(values.mean()), 2),
        "workload_standard_deviation": round(
            float(values.std(ddof=0)), 3
        ),
    }


original_workload = workload_statistics(
    "original_final_pending"
)

optimized_workload = workload_statistics(
    "optimized_final_pending"
)


# --------------------------------------------------
# 7. ASSIGNMENT AND EXPERIENCE METRICS
# --------------------------------------------------

original_assigned = (
    comparison["original_team_id"] != "REVIEW_REQUIRED"
).sum()

optimized_assigned = (
    comparison["optimized_team_id"] != "REVIEW_REQUIRED"
).sum()

original_experience = pd.to_numeric(
    comparison["original_charger_experience"],
    errors="coerce",
).mean()

optimized_experience = pd.to_numeric(
    comparison["optimized_charger_experience"],
    errors="coerce",
).mean()

team_changes = int(comparison["team_changed"].sum())

capacity_violations_original = int(
    (~team_comparison["original_within_capacity"]).sum()
)

capacity_violations_optimized = int(
    (~team_comparison["optimized_within_capacity"]).sum()
)


# --------------------------------------------------
# 8. BUILD SUMMARY
# --------------------------------------------------

summary = {
    "total_work_orders": int(len(comparison)),

    "original_method": {
        "assigned_work_orders": int(original_assigned),
        "assignment_rate_percent": round(
            original_assigned / len(comparison) * 100, 2
        ),
        "distance": original_distance,
        "workload": original_workload,
        "average_charger_specific_experience": round(
            float(original_experience), 2
        ),
        "teams_exceeding_capacity": capacity_violations_original,
    },

    "optimized_method": {
        "assigned_work_orders": int(optimized_assigned),
        "assignment_rate_percent": round(
            optimized_assigned / len(comparison) * 100, 2
        ),
        "distance": optimized_distance,
        "workload": optimized_workload,
        "average_charger_specific_experience": round(
            float(optimized_experience), 2
        ),
        "teams_exceeding_capacity": capacity_violations_optimized,
    },

    "comparison": {
        "work_orders_with_changed_team": team_changes,
        "work_orders_with_same_team": int(
            len(comparison) - team_changes
        ),
        "distance_saving_km_on_known_pairs": round(
            float(distance_saving), 2
        ),
        "distance_improvement_percent": (
            distance_improvement_percent
        ),
        "average_charger_experience_difference": round(
            float(optimized_experience - original_experience),
            2,
        ),
    },

    "limitations": [
        "Distances are approximate straight-line estimates.",
        "The duration model uses synthetic training data.",
        "Assignment scores from the two methods use different formulas and are not directly comparable.",
        "Workload is calculated using starting pending jobs plus new assignments.",
        "This evaluation does not measure real-world assignment accuracy."
    ]
}


# --------------------------------------------------
# 9. SAVE OUTPUT FILES
# --------------------------------------------------

comparison_path = OUTPUT_DIR / "assignment_comparison_details.csv"
team_path = OUTPUT_DIR / "team_workload_comparison.csv"
summary_path = OUTPUT_DIR / "assignment_comparison_summary.json"
excel_path = OUTPUT_DIR / "assignment_comparison.xlsx"

comparison.to_csv(comparison_path, index=False)
team_comparison.to_csv(team_path, index=False)

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

with pd.ExcelWriter(excel_path) as writer:
    comparison.to_excel(
        writer, sheet_name="Work Order Comparison", index=False
    )
    team_comparison.to_excel(
        writer, sheet_name="Team Workload", index=False
    )
    pd.DataFrame([
        {"metric": "Total work orders", "value": len(comparison)},
        {"metric": "Original assigned", "value": original_assigned},
        {"metric": "Optimized assigned", "value": optimized_assigned},
        {"metric": "Teams changed", "value": team_changes},
        {"metric": "Distance saved (km)", "value": round(float(distance_saving), 2)},
        {"metric": "Distance improvement (%)", "value": distance_improvement_percent},
    ]).to_excel(writer, sheet_name="Summary", index=False)


# --------------------------------------------------
# 10. PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("ASSIGNMENT METHOD COMPARISON")
print("=" * 65)

print(f"Total work orders: {len(comparison)}")
print(f"Original assignments: {original_assigned}")
print(f"Optimized assignments: {optimized_assigned}")
print(f"Work orders with changed teams: {team_changes}")

print("\nTRAVEL DISTANCE")

print(
    f"Original total: "
    f"{original_distance['total_distance_km']} km"
)
print(
    f"Optimized total: "
    f"{optimized_distance['total_distance_km']} km"
)
print(
    f"Distance saved: {distance_saving:.2f} km"
)
print(
    f"Distance improvement: "
    f"{distance_improvement_percent}%"
)

print("\nTEAM WORKLOAD COMPARISON")
print(team_comparison.to_string(index=False))

print("\nWORKLOAD BALANCE")
print("Original:", original_workload)
print("Optimized:", optimized_workload)

print("\nCHARGER-SPECIFIC EXPERIENCE")
print(
    f"Original average: {original_experience:.2f} installations"
)
print(
    f"Optimized average: {optimized_experience:.2f} installations"
)

print("\nCAPACITY CHECK")
print(
    f"Original teams exceeding capacity: "
    f"{capacity_violations_original}"
)
print(
    f"Optimized teams exceeding capacity: "
    f"{capacity_violations_optimized}"
)

print("\nFILES SAVED:")
print(comparison_path)
print(team_path)
print(summary_path)
print(excel_path)
