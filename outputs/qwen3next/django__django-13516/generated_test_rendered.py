import unittest

from django.core.management.color import no_style
from django.db import connection
from django.test import SimpleTestCase

from ..models import Person, Tag


@unittest.skipUnless(connection.vendor == 'mysql', 'MySQL tests.')
class MySQLOperationsTests(SimpleTestCase):
    def test_sql_flush(self):
        # allow_cascade doesn't change statements on MySQL.
        for allow_cascade in [False, True]:
            with self.subTest(allow_cascade=allow_cascade):
                self.assertEqual(
                    connection.ops.sql_flush(
                        no_style(),
                        [Person._meta.db_table, Tag._meta.db_table],
                        allow_cascade=allow_cascade,
                    ),
                    [
                        'SET FOREIGN_KEY_CHECKS = 0;',
                        'DELETE FROM `backends_person`;',
                        'DELETE FROM `backends_tag`;',
                        'SET FOREIGN_KEY_CHECKS = 1;',
                    ],
                )

    def test_sql_flush_sequences(self):
        # allow_cascade doesn't change statements on MySQL.
        for allow_cascade in [False, True]:
            with self.subTest(allow_cascade=allow_cascade):
                self.assertEqual(
                    connection.ops.sql_flush(
                        no_style(),
                        [Person._meta.db_table, Tag._meta.db_table],
                        reset_sequences=True,
                        allow_cascade=allow_cascade,
                    ),
                    [
                        'SET FOREIGN_KEY_CHECKS = 0;',
                        'TRUNCATE `backends_person`;',
                        'TRUNCATE `backends_tag`;',
                        'SET FOREIGN_KEY_CHECKS = 1;',
                    ],
                )

import re
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from io import StringIO

class TestFlushCommand(TestCase):
    def test_flush_command_output_repro(self):
        out = StringIO()
        err = StringIO()
        call_command('flush', stdout=out, stderr=err, interactive=False)
        self.assertIn('Applying', out.getvalue())
        self.assertNotIn('Applying', err.getvalue())
