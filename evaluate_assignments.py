"""
Operational evaluation for the EV charger assignment engine.

This evaluates the generated recommendations against operational constraints.
It does NOT calculate prediction accuracy because verified actual team choices
are not available.

Expected files:
    data/teams_with_charger_experience.csv
    outputs/assignment_report.csv

Outputs:
    outputs/evaluation_summary.json
    outputs/operational_evaluation_details.csv
"""

from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

REPORT_PATH = OUTPUT_DIR / "assignment_report.csv"
TEAMS_PATH = DATA_DIR / "teams_with_charger_experience.csv"

MAX_PENDING_JOBS = 5

REQUIRED_REPORT_COLUMNS = {
    "Work Order Number",
    "Charger Type",
    "Recommended Team ID",
    "Distance (km)",
    "Predicted Duration (hours)",
    "Assignment Status",
    "Eligible Team Count",
}

REQUIRED_TEAM_COLUMNS = {
    "team_id",
    "charger_types",
    "available",
    "completed_3_3KW",
    "completed_7_2KW",
    "completed_7_5KW",
}


def parse_charger_kw(value):
    """Convert values such as '7.2KW' to 7.2."""
    text = str(value).strip().lower().replace(" ", "").replace("kw", "")
    try:
        return round(float(text), 2)
    except (TypeError, ValueError):
        return None


def parse_supported_types(value):
    parts = str(value).replace(";", ",").split(",")
    return {
        kw for part in parts
        if (kw := parse_charger_kw(part)) is not None
    }


def parse_available(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_pending_from_reasons(value):
    """Extract the pending-job count recorded at recommendation time."""
    match = re.search(r"(\d+)\s+pending job", str(value), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def require_columns(frame, required, label):
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Assignment report not found: {REPORT_PATH}\n"
            "Run python assignment_engine.py first."
        )

    if not TEAMS_PATH.exists():
        raise FileNotFoundError(
            f"Team data not found: {TEAMS_PATH}\n"
            "Place teams_with_charger_experience.csv in the data folder."
        )

    report = pd.read_csv(REPORT_PATH)
    teams = pd.read_csv(TEAMS_PATH)

    require_columns(report, REQUIRED_REPORT_COLUMNS, "Assignment report")
    require_columns(teams, REQUIRED_TEAM_COLUMNS, "Team dataset")

    if report.empty:
        raise ValueError("Assignment report contains no work orders.")

    teams_by_id = {
        str(row["team_id"]).strip(): row
        for _, row in teams.iterrows()
    }

    checks = []
    for _, row in report.iterrows():
        work_order = str(row["Work Order Number"])
        team_id = str(row["Recommended Team ID"]).strip()
        charger_kw = parse_charger_kw(row["Charger Type"])
        status = str(row["Assignment Status"]).strip().lower()

        has_recommendation = bool(team_id) and team_id.lower() != "nan"
        team_exists = has_recommendation and team_id in teams_by_id

        supported = None
        available = None
        pending = None
        workload_ok = None
        charger_experience = None

        reasons = row.get("Recommendation Reasons", "")
        pending = parse_pending_from_reasons(reasons)

        if team_exists:
            team = teams_by_id[team_id]
            supported = (
                charger_kw in parse_supported_types(team["charger_types"])
                if charger_kw is not None else False
            )
            available = parse_available(team["available"])
            workload_ok = pending < MAX_PENDING_JOBS if pending is not None else None

            experience_column = None
            if charger_kw == 3.3:
                experience_column = "completed_3_3KW"
            elif charger_kw == 7.2:
                experience_column = "completed_7_2KW"
            elif charger_kw == 7.5:
                experience_column = "completed_7_5KW"

            if experience_column:
                value = pd.to_numeric(team[experience_column], errors="coerce")
                charger_experience = (
                    int(value) if pd.notna(value) else None
                )

        checks.append({
            "Work Order Number": work_order,
            "Charger Type": row["Charger Type"],
            "Recommended Team ID": team_id if has_recommendation else "",
            "Team Exists": team_exists,
            "Charger Compatible": supported,
            "Team Available In Dataset": available,
            "Pending Jobs At Recommendation": pending,
            "Below Workload Limit": workload_ok,
            "Charger-Specific Completed Installs": charger_experience,
            "Distance (km)": row["Distance (km)"],
            "Predicted Duration (hours)": row["Predicted Duration (hours)"],
            "Assignment Status": row["Assignment Status"],
            "Eligible Team Count": row["Eligible Team Count"],
        })

    detail = pd.DataFrame(checks)

    total = len(detail)
    recommended = int(detail["Recommended Team ID"].astype(str).str.len().gt(0).sum())
    no_recommendation = total - recommended

    def count_true(column):
        return int(detail[column].eq(True).sum())

    # A violation is counted only when the relevant check could be performed.
    compatibility_checked = detail["Charger Compatible"].notna()
    availability_checked = detail["Team Available In Dataset"].notna()
    workload_checked = detail["Below Workload Limit"].notna()

    compatible_count = count_true("Charger Compatible")
    available_count = count_true("Team Available In Dataset")
    workload_ok_count = count_true("Below Workload Limit")

    distances = pd.to_numeric(detail["Distance (km)"], errors="coerce")
    durations = pd.to_numeric(
        detail["Predicted Duration (hours)"], errors="coerce"
    )
    eligible_counts = pd.to_numeric(
        detail["Eligible Team Count"], errors="coerce"
    )

    team_distribution = (
        detail.loc[detail["Recommended Team ID"].ne("")]
        ["Recommended Team ID"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    charger_distribution = (
        detail["Charger Type"].astype(str).value_counts().sort_index().to_dict()
    )

    summary = {
        "evaluation_type": "Operational checks; not actual assignment accuracy",
        "total_work_orders": total,
        "work_orders_with_recommendation": recommended,
        "work_orders_without_recommendation": no_recommendation,
        "recommendation_rate_percent": round(
            recommended / total * 100, 2
        ) if total else None,
        "checks": {
            "recommended_team_exists_count": count_true("Team Exists"),
            "charger_compatibility_pass_count": compatible_count,
            "charger_compatibility_checked_count": int(compatibility_checked.sum()),
            "team_available_pass_count": available_count,
            "team_availability_checked_count": int(availability_checked.sum()),
            "workload_limit_pass_count": workload_ok_count,
            "workload_limit_checked_count": int(workload_checked.sum()),
            "workload_limit": MAX_PENDING_JOBS,
        },
        "distance_km": {
            "known_count": int(distances.notna().sum()),
            "unknown_count": int(distances.isna().sum()),
            "average": round(float(distances.mean()), 2) if distances.notna().any() else None,
            "minimum": round(float(distances.min()), 2) if distances.notna().any() else None,
            "maximum": round(float(distances.max()), 2) if distances.notna().any() else None,
            "note": "Straight-line Haversine distance, not road distance.",
        },
        "predicted_duration_hours": {
            "known_count": int(durations.notna().sum()),
            "average": round(float(durations.mean()), 2) if durations.notna().any() else None,
            "minimum": round(float(durations.min()), 2) if durations.notna().any() else None,
            "maximum": round(float(durations.max()), 2) if durations.notna().any() else None,
            "note": "Predictions come from a model trained on synthetic demonstration data.",
        },
        "eligible_team_count": {
            "average": round(float(eligible_counts.mean()), 2) if eligible_counts.notna().any() else None,
            "minimum": int(eligible_counts.min()) if eligible_counts.notna().any() else None,
        },
        "recommendations_by_team": team_distribution,
        "work_orders_by_charger_type": charger_distribution,
        "complexity_unknown_count": (
            int(detail["Assignment Status"].notna().sum())
            if "Complexity Used" not in report.columns
            else int(report["Complexity Used"].astype(str).str.lower().eq("unknown").sum())
        ),
        "limitations": [
            "No actual team-selection labels were supplied, so accuracy, precision, recall and F1 are not calculated.",
            "Availability is checked against the current team CSV; the report does not preserve a timestamped availability history.",
            "Workload is checked using the pending-job count recorded in the recommendation reason.",
            "Distances are approximate straight-line distances, not driving distances.",
            "Duration predictions use synthetic training data and require validation against real installation records.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details_path = OUTPUT_DIR / "operational_evaluation_details.csv"
    summary_path = OUTPUT_DIR / "evaluation_summary.json"

    detail.to_csv(details_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== OPERATIONAL ASSIGNMENT EVALUATION ===")
    print(f"Work orders processed: {total}")
    print(f"Work orders with recommendation: {recommended}")
    print(f"Work orders without recommendation: {no_recommendation}")
    print(f"Recommendation rate: {summary['recommendation_rate_percent']}%")
    print("\n=== CONSTRAINT CHECKS ===")
    print(
        f"Recommended team exists: "
        f"{summary['checks']['recommended_team_exists_count']}/{recommended}"
    )
    print(
        f"Charger compatibility passed: "
        f"{compatible_count}/{int(compatibility_checked.sum())} checked"
    )
    print(
        f"Team available in dataset: "
        f"{available_count}/{int(availability_checked.sum())} checked"
    )
    print(
        f"Below workload limit: "
        f"{workload_ok_count}/{int(workload_checked.sum())} checked"
    )
    print("\n=== DISTANCE AND DURATION ===")
    print(f"Average known distance: {summary['distance_km']['average']} km")
    print(
        "Average predicted duration: "
        f"{summary['predicted_duration_hours']['average']} hours"
    )
    print("\n=== RECOMMENDATIONS BY TEAM ===")
    for team_id, count in team_distribution.items():
        print(f"{team_id}: {count}")
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {details_path}")
    print("\nNOTE: These are operational checks, not ML accuracy metrics.")


if __name__ == "__main__":
    main()
