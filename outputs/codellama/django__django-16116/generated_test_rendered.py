from django.core.management.base import BaseCommand


class Command(BaseCommand):
    requires_system_checks = []

    def handle(self, *args, **options):
        pass

from django.test import TestCase, SimpleTestCase
import re
from django.core.management import call_command
from io import StringIO

class TestMakemigrationsCheckBehavior(TestCase):
    def test_makemigrations_check_exits_without_creating_migrations_repro(self):
        out = StringIO()
        call_command('makemigrations', check=True, stdout=out)
        self.assertNotIn('Creating migration', out.getvalue())
