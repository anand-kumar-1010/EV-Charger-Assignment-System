import csv
import io
import json
from pathlib import Path

import joblib
import pandas as pd

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST

import assignment_engine as engine
from .models import AssignmentApproval


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(settings.BASE_DIR)
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# ==========================================================
# COMMON HELPERS
# ==========================================================

def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def read_csv_file(file_path):
    if not file_path.exists():
        return []

    with file_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def get_work_orders():
    return read_csv_file(DATA_DIR / "installation_requests.csv")


def get_teams():
    return read_csv_file(DATA_DIR / "teams.csv")


def find_column(df, *possible_names):
    """Find a column without depending on capitalization or underscores."""
    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in df.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]

    return None


def normalize_orders(df):
    """Normalize uploaded CSV/Excel rows to consistent field names."""

    order_col = find_column(
        df, "Work Order Number", "work_order_number", "work_order"
    )
    charger_col = find_column(
        df, "Charger Type", "charger_type"
    )
    address_col = find_column(
        df, "Customer Address", "customer_address", "address"
    )
    date_col = find_column(
        df, "Installation Date", "installation_date"
    )
    priority_col = find_column(df, "Priority")
    notes_col = find_column(df, "Notes")

    if not order_col or not charger_col or not address_col:
        raise ValueError(
            "The uploaded file must contain these columns: "
            "Work Order Number, Charger Type, Customer Address."
        )

    orders = []

    for _, row in df.fillna("").iterrows():
        number = clean_value(row[order_col])
        charger = clean_value(row[charger_col])
        address = clean_value(row[address_col])

        if not number or not charger or not address:
            continue

        orders.append({
            "work_order_number": number,
            "charger_type": charger,
            "customer_address": address,
            "installation_date": (
                clean_value(row[date_col]) if date_col else ""
            ),
            "priority": (
                clean_value(row[priority_col])
                if priority_col else ""
            ),
            "notes": (
                clean_value(row[notes_col])
                if notes_col else ""
            ),
        })

    if not orders:
        raise ValueError(
            "No valid work orders were found in the uploaded file."
        )

    return orders


def load_assignment_resources():
    """Load team data and the trained duration model."""

    teams_path = DATA_DIR / "teams.csv"
    model_path = MODEL_DIR / "duration_model.joblib"
    metrics_path = MODEL_DIR / "model_metrics.json"

    if not teams_path.exists():
        raise FileNotFoundError("data/teams.csv was not found.")

    if not model_path.exists():
        raise FileNotFoundError(
            "models/duration_model.joblib was not found. "
            "Run python train_models.py first."
        )

    teams_df = pd.read_csv(teams_path).fillna("")
    model = joblib.load(model_path)

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

    required_columns = {
        "team_id",
        "available",
        "charger_types",
        "pending_jobs",
        "base_location",
        "experience_years",
        "completed_installations",
    }

    missing = required_columns - set(teams_df.columns)

    if missing:
        raise ValueError(
            "teams.csv is missing these columns: "
            + ", ".join(sorted(missing))
        )

    return teams_df, model, metrics


def is_available(value):
    return clean_value(value).lower() in {
        "true", "1", "yes", "available"
    }


def calculate_team_recommendations(
    order, teams_df, model, workloads
):
    """
    Rank teams for a work order.

    Fully eligible teams are preferred. If no team qualifies,
    return a fallback candidate with a warning instead of an
    empty recommendation list.
    """

    charger_type = order["charger_type"]
    address = order["customer_address"]

    required_kw = engine.charger_kw(charger_type)

    if required_kw is None:
        return [], (
            f"Unrecognized charger type: {charger_type}. "
            "Please check the uploaded file."
        )

    customer_coords = engine.find_coordinates(address)

    eligible = []
    fallback = []

    max_pending = getattr(engine, "MAX_PENDING_JOBS", 5)
    distance_weight = getattr(engine, "DISTANCE_WEIGHT", 0.5)
    workload_weight = getattr(engine, "WORKLOAD_WEIGHT", 5.0)
    experience_weight = getattr(engine, "EXPERIENCE_WEIGHT", 1.0)
    completed_weight = getattr(
        engine, "COMPLETED_BONUS_WEIGHT", 0.02
    )

    for _, team in teams_df.iterrows():
        team_id = clean_value(team["team_id"])
        base = clean_value(team["base_location"])

        try:
            pending = int(float(team["pending_jobs"] or 0))
            experience = float(team["experience_years"] or 0)
            completed = int(
                float(team["completed_installations"] or 0)
            )
        except (TypeError, ValueError):
            continue

        available = is_available(team["available"])

        supported_types = engine.supported_types(
            team["charger_types"]
        )
        supports_charger = required_kw in supported_types

        team_coords = engine.find_coordinates(base)

        distance = None
        if team_coords and customer_coords:
            distance = engine.haversine_km(
                team_coords, customer_coords
            )

        distance_component = (
            distance * distance_weight
            if distance is not None else 25.0
        )

        score = (
            pending * workload_weight
            + distance_component
            - experience * experience_weight
            - min(completed, 100) * completed_weight
        )

        reasons = []

        if not available:
            reasons.append("Team marked unavailable")

        if not supports_charger:
            reasons.append("Charger type not supported")

        if pending >= max_pending:
            reasons.append("Workload limit reached")

        item = {
            "team_id": team_id,
            "base": base,
            "pending": pending,
            "experience": experience,
            "completed": completed,
            "distance": (
                round(float(distance), 2)
                if distance is not None else None
            ),
            "score": round(score, 2),
            "available": available,
            "supports_charger": supports_charger,
            "fallback": bool(reasons),
            "reason": "; ".join(reasons),
            "duration": None,
        }

        # Use the trained model for estimated installation duration.
        # These feature names must match the training pipeline.
        features = pd.DataFrame([{
            "charger_kw": required_kw,
            "complexity": "medium",
            "distance_km": (
                float(distance) if distance is not None else 0.0
            ),
            "team_experience_years": experience,
            "team_completed_installations": completed,
            "team_pending_jobs": pending,
        }])

        try:
            item["duration"] = round(
                max(0.0, float(model.predict(features)[0])), 2
            )
        except Exception:
            # Keep the recommendation visible if prediction fails.
            item["duration"] = None

        if (
            available
            and supports_charger
            and pending < max_pending
        ):
            eligible.append(item)
        else:
            fallback.append(item)

    eligible.sort(key=lambda item: item["score"])
    fallback.sort(key=lambda item: item["score"])

    if eligible:
        return eligible, None

    if fallback:
        return fallback, (
            "No team meets all assignment requirements. "
            "The ranked alternatives are shown for supervisor review. "
            "Do not assign an unavailable or incompatible team "
            "without resolving the restriction."
        )

    return [], "No teams are registered in data/teams.csv."


def generate_assignment_report(orders, teams_df, model):
    """Generate one report row for every uploaded work order."""

    report_rows = []

    # Carry workload changes across the batch so that the same
    # starting workload is not assumed for every work order.
    workloads = {
        clean_value(row["team_id"]): int(
            float(row["pending_jobs"] or 0)
        )
        for _, row in teams_df.iterrows()
    }

    for order in orders:
        recommendations, warning = calculate_team_recommendations(
            order, teams_df, model, workloads
        )

        best = recommendations[0] if recommendations else None

        if best:
            team_name = best["team_id"]
            status = (
                "Fallback - supervisor review required"
                if best["fallback"]
                else "Recommended - pending approval"
            )

            # Reserve workload for later orders in this uploaded batch.
            if best["available"] and best["supports_charger"]:
                workloads[team_name] = (
                    workloads.get(team_name, 0) + 1
                )

            report_rows.append({
                "work_order": order["work_order_number"],
                "charger_type": order["charger_type"],
                "address": order["customer_address"],
                "eligible_teams": ", ".join(
                    item["team_id"] for item in recommendations
                    if not item["fallback"]
                ) or "None",
                "team": team_name,
                "distance": (
                    best["distance"]
                    if best["distance"] is not None else ""
                ),
                "duration": (
                    best["duration"]
                    if best["duration"] is not None else ""
                ),
                "result": status,
                "warning": warning or best["reason"],
            })

        else:
            report_rows.append({
                "work_order": order["work_order_number"],
                "charger_type": order["charger_type"],
                "address": order["customer_address"],
                "eligible_teams": "None",
                "team": "Supervisor review",
                "distance": "",
                "duration": "",
                "result": "No team recommendation",
                "warning": warning or "Manual review required",
            })

    return report_rows


# ==========================================================
# DASHBOARD
# ==========================================================

def dashboard(request):
    return render(request, "assignment/dashboard.html")


# ==========================================================
# WORK ORDERS
# ==========================================================

def work_orders(request):
    if not request.session.get("current_upload_ready"):
        messages.info(
            request,
            "Please upload a CSV or Excel file first."
        )
        return redirect("assignment:dashboard")

    orders = request.session.get("current_work_orders", [])

    return render(
        request,
        "assignment/work_orders.html",
        {
            "orders": orders,
            "current_upload_name": request.session.get(
                "current_upload_name", ""
            ),
        },
    )


# ==========================================================
# TEAM MANAGEMENT
# ==========================================================

def team_management(request):
    return render(
        request,
        "assignment/teams.html",
        {"teams": get_teams()},
    )


# ==========================================================
# SMART ASSIGNMENT
# ==========================================================

def smart_assignment(request):
    if not request.session.get("current_upload_ready"):
        messages.info(
            request,
            "Please upload a CSV or Excel file first."
        )
        return redirect("assignment:dashboard")

    orders = request.session.get("current_work_orders", [])

    if not orders:
        messages.error(request, "No uploaded work orders were found.")
        return redirect("assignment:dashboard")

    selected_number = (
        request.POST.get("work_order")
        or request.GET.get("work_order")
    )

    selected_order = None
    recommendations = []
    error = None
    model_name = request.session.get(
        "current_model_name", "Unknown"
    )

    if selected_number:
        selected_order = next(
            (
                order for order in orders
                if str(order.get("work_order_number", ""))
                == str(selected_number)
            ),
            None,
        )

        if selected_order is None:
            error = "The selected work order was not found."
        else:
            try:
                teams_df, model, metrics = load_assignment_resources()
                model_name = metrics.get(
                    "selected_model", model_name
                )

                workloads = {
                    clean_value(row["team_id"]): int(
                        float(row["pending_jobs"] or 0)
                    )
                    for _, row in teams_df.iterrows()
                }

                recommendations, error = (
                    calculate_team_recommendations(
                        selected_order,
                        teams_df,
                        model,
                        workloads,
                    )
                )

            except Exception as exc:
                error = str(exc)

    return render(
        request,
        "assignment/assignment.html",
        {
            "orders": orders,
            "selected_order": selected_order,
            "recommendations": recommendations,
            "error": error,
            "model_name": model_name,
            "current_upload_name": request.session.get(
                "current_upload_name", ""
            ),
        },
    )


# ==========================================================
# CSV / EXCEL UPLOAD AND AUTOMATIC REPORT GENERATION
# ==========================================================

@require_POST
def upload_work_orders(request):
    uploaded_file = request.FILES.get("work_order_file")

    if not uploaded_file:
        messages.error(request, "Please choose a CSV or Excel file.")
        return redirect("assignment:dashboard")

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str).fillna("")

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(
                uploaded_file,
                dtype=str,
                engine="openpyxl",
            ).fillna("")

        elif filename.endswith(".xls"):
            df = pd.read_excel(
                uploaded_file,
                dtype=str,
            ).fillna("")

        else:
            raise ValueError(
                "Unsupported format. Upload CSV, XLSX, or XLS."
            )

        orders = normalize_orders(df)
        teams_df, model, metrics = load_assignment_resources()

        # Generate recommendations BEFORE redirecting to Reports.
        report_rows = generate_assignment_report(
            orders, teams_df, model
        )

        # Store only this upload's data in the current session.
        request.session["current_work_orders"] = orders
        request.session["current_assignment_results"] = report_rows
        request.session["current_model_name"] = metrics.get(
            "selected_model", "Unknown"
        )
        request.session["current_upload_name"] = uploaded_file.name
        request.session["current_upload_ready"] = True

        # Reset decisions for the new upload and create approval records.
        request.session["assignment_decisions"] = []
        request.session.modified = True

        for row in report_rows:
            if row["team"] == "Supervisor review":
                continue

            AssignmentApproval.objects.update_or_create(
                work_order_number=row["work_order"],
                defaults={
                    "recommended_team": row["team"],
                    "status": "PENDING",
                    "supervisor_notes": "",
                },
            )

        messages.success(
            request,
            f"Processed {len(orders)} work orders and generated "
            "the assignment report."
        )

        # Successful upload goes directly to Reports.
        return redirect("assignment:reports")

    except Exception as exc:
        messages.error(
            request,
            f"Could not process the uploaded file: {exc}"
        )
        return redirect("assignment:dashboard")


# ==========================================================
# REPORTS
# ==========================================================

def reports(request):
    rows = request.session.get("current_assignment_results", [])

    assigned_count = sum(
        1 for row in rows
        if row.get("team") not in ("Supervisor review", "", None)
    )

    review_count = len(rows) - assigned_count

    return render(
        request,
        "assignment/reports.html",
        {
            "reports": rows,
            "total_orders": len(rows),
            "assigned_orders": assigned_count,
            "review_orders": review_count,
            "current_upload_name": request.session.get(
                "current_upload_name", ""
            ),
            "model_name": request.session.get(
                "current_model_name", "Unknown"
            ),
        },
    )


# ==========================================================
# DOWNLOAD REPORT AS CSV OR EXCEL
# ==========================================================

def download_report(request, file_type):
    rows = request.session.get("current_assignment_results", [])

    if not rows:
        messages.error(
            request,
            "Upload a file and generate a report first."
        )
        return redirect("assignment:reports")

    df = pd.DataFrame(rows)

    if file_type.lower() == "csv":
        response = HttpResponse(
            df.to_csv(index=False),
            content_type="text/csv",
        )
        response["Content-Disposition"] = (
            'attachment; filename="assignment_report.csv"'
        )
        return response

    if file_type.lower() in ("xlsx", "excel"):
        buffer = io.BytesIO()

        with pd.ExcelWriter(
            buffer, engine="openpyxl"
        ) as writer:
            df.to_excel(
                writer,
                index=False,
                sheet_name="Assignments",
            )

        response = HttpResponse(
            buffer.getvalue(),
            content_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )
        response["Content-Disposition"] = (
            'attachment; filename="assignment_report.xlsx"'
        )
        return response

    raise Http404("Unsupported report format.")


# ==========================================================
# LEGACY CSV UPLOAD PAGE
# ==========================================================

def upload_csv(request):
    """
    Keep this view if your existing URL configuration still uses
    the separate upload_csv.html page.
    """
    if request.method != "POST":
        return render(
            request,
            "assignment/upload_csv.html",
            {},
        )

    uploaded_file = request.FILES.get("csv_file")

    if not uploaded_file:
        return render(
            request,
            "assignment/upload_csv.html",
            {"error": "Please select a CSV file."},
        )

    try:
        if not uploaded_file.name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported on this page.")

        df = pd.read_csv(uploaded_file, dtype=str).fillna("")
        orders = normalize_orders(df)

        teams_df, model, metrics = load_assignment_resources()
        report_rows = generate_assignment_report(
            orders, teams_df, model
        )

        request.session["current_work_orders"] = orders
        request.session["current_assignment_results"] = report_rows
        request.session["current_model_name"] = metrics.get(
            "selected_model", "Unknown"
        )
        request.session["current_upload_name"] = uploaded_file.name
        request.session["current_upload_ready"] = True
        request.session.modified = True

        messages.success(
            request,
            f"Processed {len(orders)} work orders successfully."
        )

        return redirect("assignment:reports")

    except Exception as exc:
        return render(
            request,
            "assignment/upload_csv.html",
            {"error": str(exc)},
        )


# ==========================================================
# DOWNLOAD LEGACY UPLOAD REPORT
# ==========================================================

def download_uploaded_report(request):
    rows = request.session.get("current_assignment_results", [])

    if not rows:
        raise Http404(
            "Upload a file and generate a report first."
        )

    df = pd.DataFrame(rows)

    response = HttpResponse(
        df.to_csv(index=False),
        content_type="text/csv",
    )
    response["Content-Disposition"] = (
        'attachment; filename="uploaded_assignment_report.csv"'
    )

    return response