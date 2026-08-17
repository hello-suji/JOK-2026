import unittest

from django.core.management.color import no_style
from django.db import connection
from django.test import SimpleTestCase

from ..models import Person, Tag


@unittest.skipUnless(connection.vendor == "postgresql", "PostgreSQL tests.")
class PostgreSQLOperationsTests(SimpleTestCase):
    def test_sql_flush_gate(self):
        self.assertEqual(
            connection.ops.sql_flush(
                no_style(),
                [Person._meta.db_table, Tag._meta.db_table],
            ),
            ['TRUNCATE "backends_person", "backends_tag";'],
        )

    def test_sql_flush_gate_allow_cascade_gate(self):
        self.assertEqual(
            connection.ops.sql_flush(
                no_style(),
                [Person._meta.db_table, Tag._meta.db_table],
                allow_cascade=True,
            ),
            ['TRUNCATE "backends_person", "backends_tag" CASCADE;'],
        )

    def test_sql_flush_gate_sequences_gate(self):
        self.assertEqual(
            connection.ops.sql_flush(
                no_style(),
                [Person._meta.db_table, Tag._meta.db_table],
                reset_sequences=True,
            ),
            ['TRUNCATE "backends_person", "backends_tag" RESTART IDENTITY;'],
        )

    def test_sql_flush_gate_sequences_gate_allow_cascade_gate(self):
        self.assertEqual(
            connection.ops.sql_flush(
                no_style(),
                [Person._meta.db_table, Tag._meta.db_table],
                reset_sequences=True,
                allow_cascade=True,
            ),
            ['TRUNCATE "backends_person", "backends_tag" RESTART IDENTITY CASCADE;'],
        )

from django.db import connection
from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.test import TestCase

class TestRemoveConstraint(TestCase):
    def test_remove_unique_constraint_repro(self):
        schema_editor = DatabaseSchemaEditor(connection)
        table_name = 'my_table'
        constraint_name = 'my_unique_constraint'
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE TABLE {table_name} (id SERIAL PRIMARY KEY, name VARCHAR(100) UNIQUE CONSTRAINT {constraint_name} UNIQUE)')
        schema_editor.remove_constraint(table_name, constraint_name)
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT conname FROM pg_constraint WHERE conname = %s', [constraint_name])
            self.assertFalse(cursor.fetchone(), 'The constraint should have been removed.')
