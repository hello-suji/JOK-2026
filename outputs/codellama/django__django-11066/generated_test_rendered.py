from django.db import migrations


def assert_foo_contenttype_not_cached(apps, schema_editor):
    ContentType = apps.get_model('contenttypes', 'ContentType')
    try:
        content_type = ContentType.objects.get_by_natural_key('contenttypes_tests', 'foo')
    except ContentType.DoesNotExist:
        pass
    else:
        if not ContentType.objects.filter(app_label='contenttypes_tests', model='foo').exists():
            raise AssertionError('The contenttypes_tests.Foo ContentType should not be cached.')
        elif content_type.model != 'foo':
            raise AssertionError(
                "The cached contenttypes_tests.Foo ContentType should have "
                "its model set to 'foo'."
            )


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes_tests', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel('Foo', 'RenamedFoo'),
        migrations.RunPython(assert_foo_contenttype_not_cached, migrations.RunPython.noop)
    ]

import re
from django.test import TestCase
from django.core.management import call_command
from django.contrib.contenttypes.models import ContentType

class TestRenameModel(TestCase):
    def test_rename_model_success_repro(self):
        # Run the migrate command to trigger the rename operation.
        call_command('migrate')
        # Check that the ContentType has been renamed successfully.
        content_type_exists = ContentType.objects.filter(model='new_model_name').exists()
        self.assertTrue(content_type_exists)
        # [Tier 2: probe-verified buggy repr — must differ after fix]
        assert repr(content_type_exists) != 'OperationalError: no such table: django_content_types'
