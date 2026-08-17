import unittest

from django.core.management.color import no_style
from django.db import connection
from django.test import SimpleTestCase

from ..models import Person, Tag


@unittest.skipUnless(connection.vendor == 'mysql', 'MySQL tests.')
class MySQLOperationsTests(SimpleTestCase):
    def test_sql_flush_gate(self):
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

    def test_sql_flush_gate_sequences_gate(self):
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
from django.db.migrations.operations.models import AlterIndexTogether

class TestAlterIndexTogetherOptimization(SimpleTestCase):
    def test_optimized_alter_index_together_repro(self):
        initial_options = [('field1', 'field2')]
        new_options = [('field3', 'field4')]
        operation = AlterIndexTogether('my_model', new_options, initial_options)
        self.assertEqual(len(operation.operations), 1)
