"""
Indirection layer for PostgreSQL-specific fields, so the tests don't fail when
run with a backend other than PostgreSQL.
"""
import enum

from django.db import models

try:
    from django.contrib.postgres.fields import (
        ArrayField, BigIntegerRangeField, CICharField, CIEmailField,
        CITextField, DateRangeField, DateTimeRangeField, DecimalRangeField,
        HStoreField, IntegerRangeField,
    )
    from django.contrib.postgres.search import SearchVectorField
except ImportError:
    class DummyArrayField(models.Field):
        def __init__(self, base_field, size=None, **kwargs):
            super().__init__(**kwargs)

        def deconstruct(self):
            name, path, args, kwargs = super().deconstruct()
            kwargs.update({
                'base_field': '',
                'size': 1,
            })
            return name, path, args, kwargs

    ArrayField = DummyArrayField
    BigIntegerRangeField = models.Field
    CICharField = models.Field
    CIEmailField = models.Field
    CITextField = models.Field
    DateRangeField = models.Field
    DateTimeRangeField = models.Field
    DecimalRangeField = models.Field
    HStoreField = models.Field
    IntegerRangeField = models.Field
    SearchVectorField = models.Field


class EnumField(models.CharField):
    def get_prep_value(self, value):
        return value.value if isinstance(value, enum.Enum) else value

from django.test import TestCase, SimpleTestCase
from django.db.models.constraints import UniqueConstraint

class TestUniqueConstraint(TestCase):
    def test_unique_constraint_sql_repro(self):
        constraint = UniqueConstraint(fields=['username'], name='user_username_unq')
        sql = constraint.create_sql(model=None, schema_editor=None)
        self.assertIn('UNIQUE', sql)
        self.assertIn('username', sql)
