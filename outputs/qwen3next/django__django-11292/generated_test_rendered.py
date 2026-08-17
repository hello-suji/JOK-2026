

from django.test import TestCase, SimpleTestCase
import re
from django.core.management import call_command

class TestManagementCommandSkipChecks(TestCase):
    def test_skip_checks_execution_repro(self):
        # Mocking a command that would perform checks
        class MockCommand:
            checks_executed = True

            def execute(self, *args, **options):
                if options.get('skip_checks'):
                    self.checks_executed = False

        # Register the mock command
        mock_command = MockCommand()
        call_command('check_command', skip_checks=True, command=mock_command)

        # Verify that checks were not executed
        self.assertFalse(mock_command.checks_executed)
