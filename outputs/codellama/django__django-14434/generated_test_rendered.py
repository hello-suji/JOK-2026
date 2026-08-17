from django.apps.registry import Apps
from django.db import models

# We're testing app registry presence on load, so this is handy.

new_apps = Apps(['apps'])


class TotallyNormal(models.Model):
    name = models.CharField(max_length=255)


class SoAlternative(models.Model):
    name = models.CharField(max_length=255)

    class Meta:
        apps = new_apps

from django.test import TestCase, SimpleTestCase
from django.db.models import Table, Index, CharField

class TestUniqueSQLCreation(TestCase):
    def test_create_unique_sql_with_table_instance_repro(self):
        table = Table('my_table')
        column = CharField()
        index = Index(fields=['name'], name='unique_name')
        sql_creation = index.create_sql(table, None)
        self.assertIn('UNIQUE', sql_creation)
