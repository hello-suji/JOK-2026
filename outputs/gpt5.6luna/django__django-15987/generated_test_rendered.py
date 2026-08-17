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

from django.test import TestCase, SimpleTestCase
import re
from pathlib import Path
from django.core.management.commands.loaddata import Command

class TestLoadDataCommand(TestCase):
    def test_duplicate_path_instances_not_detected_repro(self):
        command = Command()
        fixture_dir = Path('/path/to/fixtures')
        command.fixture_dirs = [fixture_dir, fixture_dir]
        unique_dirs = set(command.fixture_dirs)
        self.assertEqual(len(unique_dirs), len(command.fixture_dirs))
