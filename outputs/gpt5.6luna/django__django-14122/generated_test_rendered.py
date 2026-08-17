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
class TestGroupByOrdering(SimpleTestCase):
    def test_group_by_excludes_meta_ordering_repro(self):
        # Create some data
        Person.objects.create(name='Alice', age=30)
        Person.objects.create(name='Bob', age=30)
        Person.objects.create(name='Charlie', age=25)

        # Perform aggregation query
        result = Person.objects.values('age').annotate(count=models.Count('id')).order_by()

        # Assert that the GROUP BY clause does not include the ordering field 'name'
        query_str = str(result.query)
        self.assertNotIn('GROUP BY `person`.`name`', query_str)
        self.assertIn('GROUP BY `person`.`age`', query_str)
