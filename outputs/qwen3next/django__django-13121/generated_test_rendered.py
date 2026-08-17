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

from datetime import timedelta
from django.db.models import F
from django.test import TestCase
from django.contrib.auth.models import User

class TestDurationExpressions(TestCase):
    def test_duration_field_annotation_repro(self):
        user = User.objects.create_user(username='testuser')
        User.objects.filter(pk=user.pk).update(last_login=F('last_login') + timedelta(hours=2))
        annotated_users = User.objects.annotate(duration=F('last_login') - F('date_joined')).filter(pk=user.pk)
        self.assertEqual(annotated_users.count(), 1)
        duration = annotated_users.first().duration
        self.assertEqual(duration.total_seconds(), 7200.0)
