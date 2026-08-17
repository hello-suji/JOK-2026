from django.core.management import call_command
from django.test import override_settings

from .test_base import MigrationTestBase


class Tests(MigrationTestBase):
    """
    Deprecated model fields should still be usable in historic migrations.
    """
    @override_settings(MIGRATION_MODULES={"migrations": "migrations.deprecated_field_migrations"})
    def test_migrate(self):
        # Make sure no tables are created
        self.assertTableNotExists("migrations_ipaddressfield")
        # Run migration
        call_command("migrate", verbosity=0)
        # Make sure the right tables exist
        self.assertTableExists("migrations_ipaddressfield")
        # Unmigrate everything
        call_command("migrate", "migrations", "zero", verbosity=0)
        # Make sure it's all gone
        self.assertTableNotExists("migrations_ipaddressfield")

import re
from django.db import models
from django.core.files.storage import FileSystemStorage

class TestFileFieldDeconstructCallableStorage(MigrationTestBase):
    def test_filefield_deconstruct_callable_storage_repro(self):
        def get_storage():
            return FileSystemStorage(location='/tmp')
        file_field = models.FileField(storage=get_storage)
        name, path, args, kwargs = file_field.deconstruct()
        self.assertIn('storage', kwargs)
        self.assertTrue(callable(kwargs['storage']))
