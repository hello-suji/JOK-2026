from django.core.management import BaseCommand


class Command(BaseCommand):

    help = 'Test suppress base options command.'
    requires_system_checks = []
    suppressed_base_arguments = {
        '-v',
        '--traceback',
        '--settings',
        '--pythonpath',
        '--no-color',
        '--force-color',
        '--version',
        'file',
    }

    def add_arguments(self, parser):
        super().add_arguments(parser)
        self.add_base_argument(parser, 'file', nargs='?', help='input file')

    def handle(self, *labels, **options):
        print('EXECUTE:SuppressBaseOptionsCommand options=%s' % sorted(options.items()))

from django.test import TestCase, SimpleTestCase
import re
from django.core.management.commands.inspectdb import Command

class TestInspectDBCommand(TestCase):
    def test_foreign_key_to_specific_field_repro(self):
        command = Command()
        # Simulate the database schema as per the issue description
        # Assuming inspectdb generates models based on the provided schema
        # We need to assert that the generated model for 'bar' has a ForeignKey to 'foo(other_id)'
        output = command.handle()
        # Expected model definition for 'bar' with ForeignKey to 'foo(other_id)'
        expected_output = "other_id = models.ForeignKey('Foo', db_column='other_id', on_delete=models.CASCADE)"
        self.assertIn(expected_output, output)
