"""EV charger team recommendation engine.

Recommendations are rule-based and require supervisor approval.
The duration model is trained on synthetic demonstration data.
Team capabilities and coordinates are illustrative; verify before real use.
"""

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND POLICY SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MAX_PENDING_JOBS = 5

# Lower total score is better. These are demo policy weights, not ML-trained values.
WORKLOAD_WEIGHT = 6.0
DISTANCE_WEIGHT = 0.6
EXPERIENCE_WEIGHT = 1.0
COMPLETED_BONUS_WEIGHT = 0.02
CHARGER_EXPERIENCE_BONUS_MAX = 5.0
CHARGER_EXPERIENCE_COUNT_CAP = 50
UNKNOWN_DISTANCE_PENALTY = 25.0

TOP_N = 3

PLACE_COORDINATES = {
    "shivarampally": (17.3167, 78.4020),
    "attapur": (17.3687, 78.4294),
    "bachupally": (17.5480, 78.3670),
    "secunderabad": (17.4399, 78.4983),
    "lb nagar": (17.3457, 78.5522),
    "kukatpally": (17.4849, 78.4138),
    "mehdipatnam": (17.3950, 78.4390),
    "miyapur": (17.4968, 78.3580),
    "dilsukhnagar": (17.3688, 78.5247),
    "begumpet": (17.4435, 78.4670),
    "manikonda": (17.4050, 78.3830),
    "kompally": (17.5375, 78.4850),
    "uppal": (17.4058, 78.5591),
    "tolichowki": (17.4000, 78.4130),
    "shamshabad": (17.2020, 78.3950),
    "ameerpet": (17.4375, 78.4483),
    "nizampet": (17.5175, 78.3850),
    "kothapet": (17.3690, 78.5480),
    "alwal": (17.5000, 78.5080),
}


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def charger_kw(value):
    """Convert values such as '7.2KW' or 7.2 into a numeric kW value."""
    text = str(value).strip().lower().replace(" ", "").replace("kw", "")
    try:
        return round(float(text), 2)
    except (TypeError, ValueError):
        return None


def supported_types(value):
    """Parse a team's comma- or semicolon-separated charger types."""
    values = str(value).replace(";", ",").split(",")
    return {kw for item in values if (kw := charger_kw(item)) is not None}


def find_coordinates(place):
    """Return approximate demo coordinates for a known locality."""
    normalized = str(place).lower().strip()
    for name, coords in PLACE_COORDINATES.items():
        if name in normalized:
            return coords
    return None


def haversine_km(point_a, point_b):
    """Calculate straight-line distance in kilometres."""
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(min(1.0, a)))


def parse_complexity(request):
    """Use supplied complexity; do not silently invent a value."""
    if "Complexity" not in request.index:
        return "unknown", "Complexity column missing from request file"

    value = request.get("Complexity")
    if pd.isna(value) or not str(value).strip():
        return "unknown", "Complexity is blank"

    return str(value).strip().lower(), ""


def validate_columns(frame, required, label):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# --------------------------------------------------
# 3. MAIN ASSIGNMENT PROCESS
# --------------------------------------------------

def main():
    requests_path = DATA / "installation_requests.csv"
    teams_path = DATA / "teams_with_charger_experience.csv"
    model_path = MODEL_DIR / "duration_model.joblib"
    metrics_path = MODEL_DIR / "model_metrics.json"

    for path in (requests_path, teams_path, model_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    requests = pd.read_csv(requests_path)
    teams = pd.read_csv(teams_path)

    validate_columns(
        requests,
        ["Work Order Number", "Case Number", "Charger Type", "Customer Address"],
        "Installation requests",
    )
    validate_columns(
        teams,
        [
            "team_id", "team_name", "base_location", "charger_types",
            "experience_years", "completed_installations",
            "pending_jobs", "available",
            "completed_3_3KW", "completed_7_2KW", "completed_7_5KW",
        ],
        "Teams",
    )

    if requests.empty:
        raise ValueError("No installation requests found.")
    if teams.empty:
        raise ValueError("No teams found.")

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Validate that the duration model expects the feature names this engine supplies.
    expected_features = list(getattr(model, "feature_names_in_", []))
    required_model_features = {
        "charger_kw",
        "complexity",
        "distance_km",
        "team_experience_years",
        "team_completed_installations",
        "team_pending_jobs",
    }
    if expected_features and not required_model_features.issubset(set(expected_features)):
        raise ValueError(
            "The saved duration model does not contain the expected features. "
            "Please retrain it using the updated train_models.py."
        )

    # Track simulated workload changes while processing requests in this run.
    workload = {}
    for _, team in teams.iterrows():
        workload[str(team["team_id"])] = int(team["pending_jobs"])

    print("\n=== EV CHARGER TEAM ASSIGNMENT SYSTEM ===")
    print("Duration model:", metrics.get("selected_model", "Unknown"))
    print("WARNING: duration model uses synthetic demonstration data.")
    print("WARNING: team capabilities and coordinates are illustrative.")
    print("Distance is straight-line Haversine distance, not road distance.")
    print(f"Maximum pending jobs per team: {MAX_PENDING_JOBS}")
    print("Recommendations require supervisor approval.\n")

    report_rows = []

    for _, req in requests.iterrows():
        required_kw = charger_kw(req["Charger Type"])
        customer_coords = find_coordinates(req["Customer Address"])
        complexity, complexity_note = parse_complexity(req)

        eligible = []

        for _, team in teams.iterrows():
            team_id = str(team["team_id"])
            available = str(team["available"]).strip().lower() in {
                "true", "1", "yes", "y"
            }

            # Hard eligibility constraints.
            if not available or required_kw is None:
                continue

            if required_kw not in supported_types(team["charger_types"]):
                continue

            pending = workload[team_id]
            if pending >= MAX_PENDING_JOBS:
                continue

            team_coords = find_coordinates(team["base_location"])
            distance = (
                haversine_km(team_coords, customer_coords)
                if team_coords is not None and customer_coords is not None
                else None
            )

            experience = float(team["experience_years"])
            completed = int(team["completed_installations"])

            # Use charger-specific completed installations only after the
            # team's charger_types capability check has passed.
            experience_column = f"completed_{str(required_kw).replace('.', '_')}KW"
            if experience_column not in teams.columns:
                raise ValueError(
                    f"Missing charger experience column: {experience_column}"
                )
            charger_completed_value = team[experience_column]
            charger_completed = (
                int(charger_completed_value)
                if pd.notna(charger_completed_value)
                else 0
            )
            charger_experience_bonus = (
                min(charger_completed, CHARGER_EXPERIENCE_COUNT_CAP)
                / CHARGER_EXPERIENCE_COUNT_CAP
                * CHARGER_EXPERIENCE_BONUS_MAX
            )

            # Missing distance is passed as NaN, allowing the model's fitted
            # numeric imputer to handle it. It is NOT incorrectly treated as 0 km.
            model_features = pd.DataFrame([{
                "charger_kw": required_kw,
                "complexity": complexity,
                "distance_km": distance if distance is not None else np.nan,
                "team_experience_years": experience,
                "team_completed_installations": completed,
                "team_pending_jobs": pending,
            }])

            # Preserve the exact feature order used during model training.
            if expected_features:
                model_features = model_features[expected_features]

            predicted_duration = max(
                0.0, float(model.predict(model_features)[0])
            )

            distance_component = (
                distance * DISTANCE_WEIGHT
                if distance is not None
                else UNKNOWN_DISTANCE_PENALTY
            )

            score = (
                pending * WORKLOAD_WEIGHT
                + distance_component
                - experience * EXPERIENCE_WEIGHT
                - min(completed, 100) * COMPLETED_BONUS_WEIGHT
                - charger_experience_bonus
            )

            reasons = [
                f"Supports {required_kw:.2f} kW charger",
                f"{pending} pending job(s)",
                f"{experience:g} years of general experience",
                f"{charger_completed} completed installations for this charger type",
            ]

            if distance is not None:
                reasons.append(f"{distance:.1f} km straight-line distance")
            else:
                reasons.append("Distance unknown; penalty applied to ranking")

            if complexity_note:
                reasons.append(complexity_note)

            eligible.append({
                "team_id": team_id,
                "team_name": str(team["team_name"]),
                "base": str(team["base_location"]),
                "pending": pending,
                "experience": experience,
                "completed": completed,
                "charger_completed": charger_completed,
                "charger_experience_bonus": charger_experience_bonus,
                "duration": predicted_duration,
                "distance": distance,
                "score": score,
                "reasons": reasons,
            })

        eligible.sort(key=lambda item: item["score"])

        print("-" * 75)
        print(
            f"Work Order: {req['Work Order Number']} | "
            f"Case: {req['Case Number']}"
        )
        print(
            f"Charger: {req['Charger Type']} | "
            f"Address: {req['Customer Address']}"
        )
        print(f"Complexity used: {complexity}")

        if not eligible:
            print("SUPERVISOR REVIEW: no available compatible team found.")
            report_rows.append({
                "Work Order Number": req["Work Order Number"],
                "Case Number": req["Case Number"],
                "Customer Address": req["Customer Address"],
                "Charger Type": req["Charger Type"],
                "Complexity Used": complexity,
                "Recommended Team ID": "",
                "Team Base Location": "",
                "Distance (km)": None,
                "Predicted Duration (hours)": None,
                "Assignment Score": None,
                "Assignment Status": "Supervisor review - no eligible team",
                "Eligible Team Count": 0,
                "Recommendation Reasons": "No available compatible team",
            })
            continue

        print(f"Eligible teams: {len(eligible)}")
        print(f"Top {min(TOP_N, len(eligible))} recommendations:")

        for rank, candidate in enumerate(eligible[:TOP_N], 1):
            distance_text = (
                f"{candidate['distance']:.1f} km"
                if candidate["distance"] is not None
                else "unknown"
            )
            print(
                f"{rank}. {candidate['team_id']} ({candidate['base']}) | "
                f"distance {distance_text} | "
                f"duration {candidate['duration']:.2f} h | "
                f"pending {candidate['pending']} | "
                f"charger-specific installs {candidate['charger_completed']} | "
                f"score {candidate['score']:.2f}"
            )

        chosen = eligible[0]
        print("Suggested team:", chosen["team_id"])
        print("Why:", "; ".join(chosen["reasons"]))
        print("Recommendation only: supervisor approval required.")

        report_rows.append({
            "Work Order Number": req["Work Order Number"],
            "Case Number": req["Case Number"],
            "Customer Address": req["Customer Address"],
            "Charger Type": req["Charger Type"],
            "Complexity Used": complexity,
            "Recommended Team ID": chosen["team_id"],
            "Charger-Specific Completed Installations": chosen["charger_completed"],
            "Charger Experience Bonus": round(chosen["charger_experience_bonus"], 2),
            "Team Base Location": chosen["base"],
            "Distance (km)": (
                round(chosen["distance"], 2)
                if chosen["distance"] is not None else None
            ),
            "Predicted Duration (hours)": round(chosen["duration"], 2),
            "Assignment Score": round(chosen["score"], 2),
            "Assignment Status": "Awaiting supervisor approval",
            "Eligible Team Count": len(eligible),
            "Recommendation Reasons": "; ".join(chosen["reasons"]),
        })

        # Simulate reserving one workload slot for subsequent requests.
        workload[chosen["team_id"]] += 1

    # --------------------------------------------------
    # 4. SAVE CSV AND EXCEL REPORTS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(report_rows)

    csv_path = OUTPUT_DIR / "assignment_report.csv"
    report.to_csv(csv_path, index=False)

    print("\n=== REPORT SAVED ===")
    print(f"CSV report: {csv_path}")

    xlsx_path = OUTPUT_DIR / "assignment_report.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            report.to_excel(writer, sheet_name="Assignments", index=False)
            worksheet = writer.sheets["Assignments"]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value or "")) for cell in column_cells
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(max_length + 2, 48)

        print(f"Excel report: {xlsx_path}")

    except (ImportError, ModuleNotFoundError) as exc:
        print("Excel report not created: openpyxl is not installed.")
        print("Install with: pip install openpyxl")
        print(f"Details: {exc}")

    print(f"Total work orders processed: {len(report)}")
    print("Workload changes are simulated in memory and reset on the next run.")


if __name__ == "__main__":
    main()
