import random
import string

from django.db import models


class MyWrapper:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.value)

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return self.value == other


class MyAutoField(models.CharField):

    def __init__(self, *args, **kwargs):
        kwargs['max_length'] = 10
        super().__init__(*args, **kwargs)

    def pre_save(self, instance, add):
        value = getattr(instance, self.attname, None)
        if not value:
            value = MyWrapper(''.join(random.sample(string.ascii_lowercase, 10)))
            setattr(instance, self.attname, value)
        return value

    def to_python(self, value):
        if not value:
            return
        if not isinstance(value, MyWrapper):
            value = MyWrapper(value)
        return value

    def from_db_value(self, value, expression, connection):
        if not value:
            return
        return MyWrapper(value)

    def get_db_prep_save(self, value, connection):
        if not value:
            return
        if isinstance(value, MyWrapper):
            return str(value)
        return value

    def get_db_prep_value(self, value, connection, prepared=False):
        if not value:
            return
        if isinstance(value, MyWrapper):
            return str(value)
        return value

from django.test import TestCase
from django.db.models import CharField, TextChoices


class MyChoice(TextChoices):
    FIRST_CHOICE = 'first', 'First Choice'
    SECOND_CHOICE = 'second', 'Second Choice'

class MyObject:
    choice_field = CharField(max_length=10, choices=MyChoice.choices)

class TestMyObjectChoices(TestCase):
    def setUp(self):
        self.obj = MyObject()
        self.obj.choice_field = MyChoice.FIRST_CHOICE

    def test_created_object_is_str_repro(self):
        self.assertIsInstance(self.obj.choice_field, str)
        self.assertEqual(self.obj.choice_field, MyChoice.FIRST_CHOICE)
