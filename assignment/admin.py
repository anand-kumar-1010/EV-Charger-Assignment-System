from django.contrib import admin
from .models import AssignmentApproval


@admin.register(AssignmentApproval)
class AssignmentApprovalAdmin(admin.ModelAdmin):
    list_display = (
        "work_order_number",
        "recommended_team",
        "status",
        "created_at",
        "updated_at",
    )

    list_filter = ("status",)
    search_fields = ("work_order_number", "recommended_team")