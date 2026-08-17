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

from django.test import TestCase, SimpleTestCase
class TestExecuteSqlFlush(SimpleTestCase):
    def test_execute_sql_flush_with_using_argument_repro(self):
        sql_list = []
        connection.ops.execute_sql_flush(using='default', sql_list=sql_list)
        # Assuming the expected behavior is that the function executes without error and sql_list remains unchanged
        self.assertEqual(sql_list, [])
