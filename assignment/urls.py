from django.urls import path
from . import views

app_name = "assignment"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),

    path("work-orders/", views.work_orders, name="work_orders"),

    path("teams/", views.team_management, name="teams"),

    path("smart-assignment/", views.smart_assignment, name="smart_assignment"),

    path("reports/", views.reports, name="reports"),

    path(
        "upload-work-orders/",
        views.upload_work_orders,
        name="upload_work_orders",
    ),

    path(
        "download-report/<str:file_type>/",
        views.download_report,
        name="download_report",
    ),
]