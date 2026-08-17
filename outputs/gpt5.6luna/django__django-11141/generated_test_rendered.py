default_app_config = 'apps.default_config_app.apps.CustomConfig'

from django.test import TestCase, SimpleTestCase
import re
from django.core.management import find_commands
from pkgutil import iter_modules
import os

class TestMigrationFileDiscovery(TestCase):
    def test_iter_modules_without_init_py_repro(self):
        # Create a temporary directory to simulate a package without __init__.py
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_package')
        os.makedirs(temp_dir, exist_ok=True)

        # Simulate a migration file inside the package
        migration_file_path = os.path.join(temp_dir, '0001_initial.py')
        with open(migration_file_path, 'w') as f:
            f.write('')

        # Add the temporary directory to sys.path to make it discoverable as a package
        import sys
        sys.path.append(temp_dir)

        try:
            # Attempt to find commands (migrations) in the package
            found_commands = find_commands(temp_dir)

            # Assert that the migration file is found
            self.assertIn('0001_initial', found_commands)
        finally:
            # Clean up the temporary directory
            sys.path.remove(temp_dir)
            os.remove(migration_file_path)
            os.rmdir(temp_dir)
