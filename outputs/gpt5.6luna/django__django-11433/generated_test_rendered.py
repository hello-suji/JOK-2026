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

from django.test import TestCase
from django.contrib.contenttypes.models import ContentType


class TestFormFieldDefaultOverwrite(TestCase):
    def test_default_overwritten_by_cleaned_data_repro(self):
        ct = ContentType.objects.first()
        form_data = {'name': 'test_name', 'content_type': ct.pk}
        form = TestContentTypeForm(data=form_data)
        form.is_valid()
        self.assertEqual(form.cleaned_data['name'], 'test_name')
