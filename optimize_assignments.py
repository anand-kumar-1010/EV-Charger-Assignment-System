
from pathlib import Path
import json

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
import joblib


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

REQUESTS_FILE = DATA_DIR / "installation_requests.csv"
TEAMS_FILE = DATA_DIR / "teams_with_charger_experience.csv"
MODEL_FILE = MODEL_DIR / "duration_model.joblib"
COORDINATES_FILE = DATA_DIR / "location_coordinates.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_PENDING_JOBS = 5

# Assignment scoring weights (lower score is better)
WORKLOAD_WEIGHT = 6.0
DISTANCE_WEIGHT = 0.6
EXPERIENCE_WEIGHT = 1.0
CHARGER_EXPERIENCE_WEIGHT = 0.05

# Integer scaling required by CP-SAT
SCORE_SCALE = 1000


# --------------------------------------------------
# 2. LOAD INPUT DATA
# --------------------------------------------------

requests = pd.read_csv(REQUESTS_FILE)
teams = pd.read_csv(TEAMS_FILE)

model = joblib.load(MODEL_FILE)

required_request_columns = {
    "Work Order Number",
    "Charger Type",
    "Customer Address",
}

required_team_columns = {
    "team_id",
    "team_name",
    "base_location",
    "experience_years",
    "completed_installations",
    "pending_jobs",
    "available",
    "charger_types",
    "completed_3_3KW",
    "completed_7_2KW",
    "completed_7_5KW",
}

# --------------------------------------------------
# LOAD LOCATION COORDINATES
# --------------------------------------------------

locations_df = pd.read_csv(COORDINATES_FILE)

# Normalize locality names to lowercase
locations_df["locality"] = (
    locations_df["locality"]
    .astype(str)
    .str.strip()
    .str.lower()
)

LOCATION_COORDINATES = {}

for _, row in locations_df.iterrows():
    locality = row["locality"]

    LOCATION_COORDINATES[locality] = (
        float(row["latitude"]),
        float(row["longitude"])
    )

print("\nLOCATION DATABASE LOADED")
print("Total locations:", len(LOCATION_COORDINATES))
print("Kukatpally:", LOCATION_COORDINATES.get("kukatpally"))
print("Shivarampally:", LOCATION_COORDINATES.get("shivarampally"))

print("\nTESTING CUSTOMER LOCATION LOOKUP")

test_address = "Kukatpally, Hyderabad"

test_locality = (
    test_address.split(",")[0].strip().lower()
)

print("Extracted locality:", test_locality)
print("Coordinates:", LOCATION_COORDINATES.get(test_locality))

missing_requests = required_request_columns - set(requests.columns)
missing_teams = required_team_columns - set(teams.columns)

if missing_requests:
    raise ValueError(f"Missing request columns: {missing_requests}")

if missing_teams:
    raise ValueError(f"Missing team columns: {missing_teams}")


# --------------------------------------------------
# 3. LOCATION COORDINATES
# --------------------------------------------------

LOCATION_COORDINATES = {
    "Shivarampally": (17.3210, 78.4290),
    "Attapur": (17.3690, 78.4300),
    "Bachupally": (17.5480, 78.3640),
    "Secunderabad": (17.4399, 78.4983),
    "LB Nagar": (17.3457, 78.5522),

    "Kukatpally": (17.4948, 78.3996),
    "Mehdipatnam": (17.3930, 78.4400),
    "Miyapur": (17.4968, 78.3630),
    "Dilsukhnagar": (17.3687, 78.5247),
    "Begumpet": (17.4430, 78.4670),
    "Manikonda": (17.4010, 78.3850),
    "Kompally": (17.5400, 78.4900),
    "Uppal": (17.4050, 78.5590),
    "Tolichowki": (17.4000, 78.4160),
    "Shamshabad": (17.2020, 78.3940),
    "Ameerpet": (17.4375, 78.4483),
    "Nizampet": (17.5160, 78.3810),
    "Kothapet": (17.3690, 78.5450),
    "Alwal": (17.5000, 78.5050),
}


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return float(
        2 * radius * np.arcsin(np.sqrt(a))
    )

def find_location_coordinates(location):
    target = str(location).strip().casefold()

    for key, coordinates in LOCATION_COORDINATES.items():
        if str(key).strip().casefold() == target:
            return coordinates

    return None

def get_distance(request, team):
    customer_address = str(request["Customer Address"]).strip()

    customer_location = customer_address.split(",")[0].strip()
    team_location = str(team["base_location"]).strip()

    customer_coords = find_location_coordinates(customer_location)
    team_coords = find_location_coordinates(team_location)

    if customer_coords is None:
        print(f"WARNING: Customer locality not found: {customer_location}")
        return None

    if team_coords is None:
        print(f"WARNING: Team locality not found: {team_location}")
        return None

    return haversine_km(
        customer_coords[0],
        customer_coords[1],
        team_coords[0],
        team_coords[1]
    )
# --------------------------------------------------
# 4. CHARGER EXPERIENCE
# --------------------------------------------------


def get_charger_experience(team, charger_kw):

    charger = normalize_charger(charger_kw)

    if charger is None:
        return 0

    if np.isclose(charger, 3.3):
        return int(team["completed_3_3KW"])

    if np.isclose(charger, 7.2):
        return int(team["completed_7_2KW"])

    if np.isclose(charger, 7.5):
        return int(team["completed_7_5KW"])

    return 0



def normalize_charger(value):
    """
    Convert charger labels such as 7.5KW or 7.5 kW
    into a consistent numeric value.
    """
    import re

    match = re.search(r"\d+(?:\.\d+)?", str(value))

    if not match:
        return None

    return float(match.group())


def supports_charger(team, charger_kw):

    requested = normalize_charger(charger_kw)

    if requested is None:
        return False

    supported_values = str(team["charger_types"]).split(",")

    for value in supported_values:
        supported = normalize_charger(value)

        if supported is not None and np.isclose(
            requested, supported
        ):
            return True

    return False
print("\nCHARGER COMPATIBILITY TEST")

for _, team in teams.iterrows():
    print(
        team["team_id"],
        {
            charger: supports_charger(team, charger)
            for charger in ["3.3KW", "7.2KW", "7.5KW"]
        }
    )

def is_available(value):
    return str(value).strip().lower() in {
        "true", "1", "yes", "available"
    }


# --------------------------------------------------
# 5. BUILD ELIGIBLE TEAM OPTIONS
# --------------------------------------------------

options = {}
candidate_details = {}

for i, request in requests.iterrows():

    request_id = request["Work Order Number"]
    charger_kw = request["Charger Type"]

    options[i] = []

    for j, team in teams.iterrows():

        if not is_available(team["available"]):
            continue

        if not supports_charger(team, charger_kw):
            continue

        pending = int(team["pending_jobs"])

        if pending >= MAX_PENDING_JOBS:
            continue

        distance = get_distance(request, team)

        # Use a penalty for unknown distance in the score.
        score_distance = (
            distance if distance is not None else 25.0
        )

        charger_experience = get_charger_experience(
            team, charger_kw
        )
        # Predict installation duration for this team and work order
        distance_for_model = (
            distance if distance is not None else np.nan
        )

        model_input = pd.DataFrame([{
            "charger_kw": normalize_charger(charger_kw),
            "complexity": "unknown",
            "distance_km": distance_for_model,
            "team_experience_years": float(team["experience_years"]),
            "team_completed_installations": float(
                team["completed_installations"]
            ),
            "team_pending_jobs": float(team["pending_jobs"]),
        }])

        predicted_duration = max(
            0.0,
            float(model.predict(model_input)[0])
        )

        experience = float(team["experience_years"])
        completed = float(team["completed_installations"])

        score = (
            pending * WORKLOAD_WEIGHT
            + score_distance * DISTANCE_WEIGHT
            - experience * EXPERIENCE_WEIGHT
            - min(charger_experience, 100)
              * CHARGER_EXPERIENCE_WEIGHT
            - min(completed, 100) * 0.02
        )

        options[i].append(j)

        candidate_details[(i, j)] = {
            "distance": distance,
            "score": score,
            "charger_experience": charger_experience,
            "predicted_duration": predicted_duration,
        }


# --------------------------------------------------
# 6. CREATE CP-SAT OPTIMIZATION MODEL
# --------------------------------------------------

cp = cp_model.CpModel()

x = {}

for i in range(len(requests)):
    for j in options[i]:
        x[(i, j)] = cp.NewBoolVar(f"assign_{i}_{j}")


# Each work order gets exactly one eligible team.
# If no eligible team exists, it is marked for review.

unassigned = {}

for i in range(len(requests)):

    if options[i]:
        cp.Add(
            sum(x[(i, j)] for j in options[i]) == 1
        )
    else:
        unassigned[i] = True


# Team capacity constraint:
# Existing pending jobs + new assignments <= 5.

for j, team in teams.iterrows():

    initial_pending = int(team["pending_jobs"])

    remaining_capacity = max(
        0,
        MAX_PENDING_JOBS - initial_pending
    )

    assigned_to_team = [
        x[(i, j)]
        for i in range(len(requests))
        if (i, j) in x
    ]

    if assigned_to_team:
        cp.Add(
            sum(assigned_to_team) <= remaining_capacity
        )



# --------------------------------------------------
# 7. MULTI-OBJECTIVE OPTIMIZATION
# --------------------------------------------------

# Configurable policy weights
DISTANCE_COST_WEIGHT = 100
DURATION_COST_WEIGHT = 200
EXPERIENCE_BONUS_WEIGHT = 10
WORKLOAD_BALANCE_WEIGHT = 500

objective_terms = []

for (i, j), variable in x.items():

    details = candidate_details[(i, j)]

    distance = details["distance"]
    duration = details["predicted_duration"]
    experience = details["charger_experience"]

    # Use a penalty for unknown distance.
    if distance is None:
        distance = 25.0

    # Lower cost is better.
    assignment_cost = (
        distance * DISTANCE_COST_WEIGHT
        + duration * DURATION_COST_WEIGHT
        - min(experience, 100) * EXPERIENCE_BONUS_WEIGHT
    )

    integer_cost = round(assignment_cost * SCORE_SCALE)

    objective_terms.append(integer_cost * variable)


# Calculate final workload for every team.
final_workloads = []

for j, team in teams.iterrows():

    initial_pending = int(team["pending_jobs"])

    new_assignments = [
        x[(i, j)]
        for i in range(len(requests))
        if (i, j) in x
    ]

    final_jobs = cp.NewIntVar(
        initial_pending,
        MAX_PENDING_JOBS,
        f"final_workload_{j}"
    )

    if new_assignments:
        cp.Add(
            final_jobs == initial_pending + sum(new_assignments)
        )
    else:
        cp.Add(final_jobs == initial_pending)

    final_workloads.append(final_jobs)


# Minimize the difference between the busiest
# and least busy teams.
max_workload = cp.NewIntVar(
    0, MAX_PENDING_JOBS, "max_workload"
)

min_workload = cp.NewIntVar(
    0, MAX_PENDING_JOBS, "min_workload"
)

cp.AddMaxEquality(max_workload, final_workloads)
cp.AddMinEquality(min_workload, final_workloads)

workload_imbalance = cp.NewIntVar(
    0, MAX_PENDING_JOBS, "workload_imbalance"
)

cp.Add(
    workload_imbalance == max_workload - min_workload
)

# Minimize combined assignment cost and workload imbalance.
cp.Minimize(
    sum(objective_terms)
    + workload_imbalance
      * WORKLOAD_BALANCE_WEIGHT
      * SCORE_SCALE
)

# --------------------------------------------------
# 8. SOLVE
# --------------------------------------------------

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8

status = solver.Solve(cp)

if status not in (
    cp_model.OPTIMAL,
    cp_model.FEASIBLE
):
    raise RuntimeError(
        "The optimizer could not find a feasible assignment plan."
    )


# --------------------------------------------------
# 9. GENERATE OPTIMIZED ASSIGNMENTS
# --------------------------------------------------

results = []

for i, request in requests.iterrows():

    assigned_team = None

    for j in options[i]:

        if solver.Value(x[(i, j)]) == 1:
            assigned_team = teams.iloc[j]
            break

    if assigned_team is None:

        results.append({
            "work_order_number": request["Work Order Number"],
        "case_number": request["Case Number"],
        "customer_address": request["Customer Address"],
        "charger_type": request["Charger Type"],
            "assigned_team_id": "REVIEW_REQUIRED",
            "assigned_team_name": "No eligible team",
            "distance_km": None,
            "charger_specific_installations": None,
            "assignment_score": None,
            "status": "Supervisor review required",
        })

        continue

    j = assigned_team.name
    details = candidate_details[(i, j)]

    results.append({
        "work_order_number": request["Work Order Number"],
        "case_number": request["Case Number"],
        "customer_address": request["Customer Address"],
        "charger_type": request["Charger Type"],
        "assigned_team_id": assigned_team["team_id"],
        "assigned_team_name": assigned_team["team_name"],
        "distance_km": (
            round(details["distance"], 2)
            if details["distance"] is not None
            else None
        ),
        "charger_specific_installations": details["charger_experience"],
        "assignment_score": round(details["score"], 3),
        "status": "Pending supervisor approval",
    })


# --------------------------------------------------
# 10. SAVE REPORTS
# --------------------------------------------------

report = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "optimized_assignment_report.csv"
excel_path = OUTPUT_DIR / "optimized_assignment_report.xlsx"

report.to_csv(csv_path, index=False)
report.to_excel(excel_path, index=False)


# --------------------------------------------------
# 11. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("GLOBAL WORK ORDER OPTIMIZATION REPORT")
print("=" * 65)

print(f"Solver status: {solver.StatusName(status)}")
print(f"Total work orders: {len(requests)}")

assigned_count = (
    report["assigned_team_id"] != "REVIEW_REQUIRED"
).sum()

print(f"Assigned work orders: {assigned_count}")
print(f"Requiring supervisor review: {len(requests) - assigned_count}")

print("\nTEAM ALLOCATION")

print(
    report[
        report["assigned_team_id"] != "REVIEW_REQUIRED"
    ]["assigned_team_name"].value_counts()
)

print("\nOPTIMIZED ASSIGNMENT DETAILS")

print(report.to_string(index=False))

print("\nSaved files:")
print(csv_path)
print(excel_path)

print("\nNote:")
print("- Assignments require supervisor approval.")
print("- Distances are approximate straight-line estimates.")
print("- Duration model uses synthetic training data.")
print("- Workload is optimized from the starting team data.")
