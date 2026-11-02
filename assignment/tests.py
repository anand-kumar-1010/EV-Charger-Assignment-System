
import pandas as pd

from django.test import TestCase, SimpleTestCase

from .models import AssignmentApproval
from .views import normalize_orders, is_available


class AssignmentApprovalTests(TestCase):

    def test_create_pending_approval(self):
        approval = AssignmentApproval.objects.create(
            work_order_number="1205101",
            recommended_team="T04"
        )

        self.assertEqual(approval.status, "PENDING")
        self.assertEqual(approval.recommended_team, "T04")

    def test_approve_assignment(self):
        approval = AssignmentApproval.objects.create(
            work_order_number="1205102",
            recommended_team="T02"
        )

        approval.status = "APPROVED"
        approval.supervisor_notes = "Approved by supervisor"
        approval.save()

        updated = AssignmentApproval.objects.get(
            work_order_number="1205102"
        )

        self.assertEqual(updated.status, "APPROVED")
        self.assertEqual(
            updated.supervisor_notes,
            "Approved by supervisor"
        )

    def test_reject_assignment(self):
        approval = AssignmentApproval.objects.create(
            work_order_number="1205103",
            recommended_team="T01"
        )

        approval.status = "REJECTED"
        approval.supervisor_notes = "Team unavailable"
        approval.save()

        self.assertEqual(
            AssignmentApproval.objects.get(
                work_order_number="1205103"
            ).status,
            "REJECTED"
        )

    def test_duplicate_work_order_not_allowed(self):
        AssignmentApproval.objects.create(
            work_order_number="1205104",
            recommended_team="T03"
        )

        with self.assertRaises(Exception):
            AssignmentApproval.objects.create(
                work_order_number="1205104",
                recommended_team="T05"
            )


class WorkOrderNormalizationTests(SimpleTestCase):

    def test_valid_work_order_csv_data(self):
        df = pd.DataFrame({
            "Work Order Number": ["1205101"],
            "Charger Type": ["7.5KW"],
            "Customer Address": ["Kukatpally, Hyderabad"]
        })

        orders = normalize_orders(df)

        self.assertEqual(len(orders), 1)
        self.assertEqual(
            orders[0]["work_order_number"],
            "1205101"
        )
        self.assertEqual(orders[0]["charger_type"], "7.5KW")

    def test_missing_required_column(self):
        df = pd.DataFrame({
            "Work Order Number": ["1205101"],
            "Charger Type": ["7.5KW"]
        })

        with self.assertRaises(ValueError):
            normalize_orders(df)

    def test_empty_rows_are_skipped(self):
        df = pd.DataFrame({
            "Work Order Number": ["1205101", ""],
            "Charger Type": ["7.5KW", ""],
            "Customer Address": [
                "Kukatpally, Hyderabad",
                ""
            ]
        })

        orders = normalize_orders(df)

        self.assertEqual(len(orders), 1)


class TeamAvailabilityTests(SimpleTestCase):

    def test_available_values(self):
        for value in ["True", "true", "1", "yes", "available"]:
            with self.subTest(value=value):
                self.assertTrue(is_available(value))

    def test_unavailable_values(self):
        for value in ["False", "0", "no", "unavailable", ""]:
            with self.subTest(value=value):
                self.assertFalse(is_available(value))

