from django.core.management import BaseCommand


class Command(BaseCommand):

    help = "Test suppress base options command."
    requires_system_checks = []
    suppressed_base_arguments = {
        "-v",
        "--traceback",
        "--settings",
        "--pythonpath",
        "--no-color",
        "--force-color",
        "--version",
        "file",
    }

    def add_arguments(self, parser):
        super().add_arguments(parser)
        self.add_base_argument(parser, "file", nargs="?", help="input file")

    def handle(self, *labels, **options):
        print("EXECUTE:SuppressBaseOptionsCommand options=%s" % sorted(options.items()))

from django.contrib import admin
from django.test import TestCase
from django.db import transaction
from django.contrib.auth.models import User
from django.contrib.admin.options import ModelAdmin
from django.contrib.contenttypes.models import ContentType

class TestChangelistViewTransactionHandling(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser('superuser', 'email@example.com', 'password')
        self.client.login(username='superuser', password='password')
        self.admin_site = admin.AdminSite()
        self.model_admin = ModelAdmin(ContentType, self.admin_site)

    def test_changelist_view_transaction_handling_repro(self):
        request = self.client.get('/admin/contenttypes/contenttype/')
        response = self.model_admin.changelist_view(request)
        self.assertTrue(transaction.get_connection().in_atomic_block)
