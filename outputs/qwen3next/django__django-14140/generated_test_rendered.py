

from django.test import TestCase, SimpleTestCase
from django.db.models import Q

class TestQObjectDeconstruction(TestCase):
    def test_single_child_q_object_deconstruction_repro(self):
        q_obj = Q(x=1)
        _, args, kwargs = q_obj.deconstruct()
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {'x': 1})
