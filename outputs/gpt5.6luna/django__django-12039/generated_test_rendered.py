import unittest

from django.db import connection
from django.test import TestCase


@unittest.skipUnless(connection.vendor == 'mysql', 'MySQL tests')
class SchemaEditorTests(TestCase):
    def test_quote_value_gate(self):
        import MySQLdb
        editor = connection.schema_editor()
        tested_values = [
            ('string', "'string'"),
            (42, '42'),
            (1.754, '1.754e0' if MySQLdb.version_info >= (1, 3, 14) else '1.754'),
            (False, b'0' if MySQLdb.version_info >= (1, 4, 0) else '0'),
        ]
        for value, expected in tested_values:
            with self.subTest(value=value):
                self.assertEqual(editor.quote_value(value), expected)

from django.test import TestCase, SimpleTestCase
from django.db.models.indexes import Index
from django.db import models

class TestIndexCreation(TestCase):
    def test_index_creation_whitespace_repro(self):
        index = Index(fields=['-name'], name='idx')
        editor = connection.schema_editor()
        sql = editor.sql_create_index(None, index)
        self.assertIn('CREATE INDEX `idx` ON `schema_author` (`name` DESC)', sql)
