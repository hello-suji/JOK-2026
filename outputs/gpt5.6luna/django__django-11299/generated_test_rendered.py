from django.core import checks
from django.db import models
from django.test import SimpleTestCase
from django.test.utils import isolate_apps


@isolate_apps('invalid_models_tests')
class DeprecatedFieldsTests(SimpleTestCase):
    def test_IPAddressField_deprecated(self):
        class IPAddressModel(models.Model):
            ip = models.IPAddressField()

        model = IPAddressModel()
        self.assertEqual(
            model.check(),
            [checks.Error(
                'IPAddressField has been removed except for support in '
                'historical migrations.',
                hint='Use GenericIPAddressField instead.',
                obj=IPAddressModel._meta.get_field('ip'),
                id='fields.E900',
            )],
        )

    def test_CommaSeparatedIntegerField_deprecated(self):
        class CommaSeparatedIntegerModel(models.Model):
            csi = models.CommaSeparatedIntegerField(max_length=64)

        model = CommaSeparatedIntegerModel()
        self.assertEqual(
            model.check(),
            [checks.Error(
                'CommaSeparatedIntegerField is removed except for support in '
                'historical migrations.',
                hint='Use CharField(validators=[validate_comma_separated_integer_list]) instead.',
                obj=CommaSeparatedIntegerModel._meta.get_field('csi'),
                id='fields.E901',
            )],
        )

from django.db.migrations import AddConstraint
from django.db.migrations.operations.models import CreateModel
from django.db.models import IntegerField, BooleanField
from django.db.models.constraints import CheckConstraint
from django.test import SimpleTestCase

class TestCheckConstraintWithOrAndClauses(SimpleTestCase):
    def test_constraint_with_or_and_clauses_repro(self):
        constraint = CheckConstraint(check=(Q(field1=True) | Q(field2=False)) & Q(field3__gte=0), name='check_constraint')
        operation = AddConstraint('MyModel', constraint)
        self.assertEqual(operation.name, 'check_constraint')
