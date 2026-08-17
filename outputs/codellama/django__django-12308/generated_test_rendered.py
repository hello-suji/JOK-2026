from django.db import models
from django.test import SimpleTestCase
from django.test.utils import isolate_apps


@isolate_apps('invalid_models_tests')
class CustomFieldTest(SimpleTestCase):

    def test_none_column(self):
        class NoColumnField(models.AutoField):
            def db_type(self, connection):
                # None indicates not to create a column in the database.
                return None

        class Model(models.Model):
            field = NoColumnField(primary_key=True, db_column="other_field")
            other_field = models.IntegerField()

        field = Model._meta.get_field('field')
        self.assertEqual(field.check(), [])

from django.test import TestCase, SimpleTestCase
from django.contrib.postgres.forms.jsonb import JSONField
from django.contrib.postgres.fields.jsonb import InvalidJSONInput
from django.contrib.admin.utils import display_for_field


class TestJSONFieldDisplay(SimpleTestCase):
    def test_jsonfield_display_as_invalid_json_repro(self):
        field = JSONField()
        value = InvalidJSONInput('{"foo": "bar"}')
        displayed_value = display_for_field(value, field)
        self.assertNotEqual(displayed_value, "{'foo': 'bar'}")
