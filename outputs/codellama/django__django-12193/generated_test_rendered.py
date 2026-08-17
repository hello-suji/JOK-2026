import pickle

from django.forms import BooleanField, ValidationError
from django.test import SimpleTestCase


class BooleanFieldTest(SimpleTestCase):

    def test_booleanfield_clean_1_gate(self):
        f = BooleanField()
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean('')
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(None)
        self.assertTrue(f.clean(True))
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(False)
        self.assertTrue(f.clean(1))
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(0)
        self.assertTrue(f.clean('Django rocks'))
        self.assertTrue(f.clean('True'))
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean('False')

    def test_booleanfield_clean_2_gate(self):
        f = BooleanField(required=False)
        self.assertIs(f.clean(''), False)
        self.assertIs(f.clean(None), False)
        self.assertIs(f.clean(True), True)
        self.assertIs(f.clean(False), False)
        self.assertIs(f.clean(1), True)
        self.assertIs(f.clean(0), False)
        self.assertIs(f.clean('1'), True)
        self.assertIs(f.clean('0'), False)
        self.assertIs(f.clean('Django rocks'), True)
        self.assertIs(f.clean('False'), False)
        self.assertIs(f.clean('false'), False)
        self.assertIs(f.clean('FaLsE'), False)

    def test_boolean_picklable_gate(self):
        self.assertIsInstance(pickle.loads(pickle.dumps(BooleanField())), BooleanField)

    def test_booleanfield_changed_gate(self):
        f = BooleanField()
        self.assertFalse(f.has_changed(None, None))
        self.assertFalse(f.has_changed(None, ''))
        self.assertFalse(f.has_changed('', None))
        self.assertFalse(f.has_changed('', ''))
        self.assertTrue(f.has_changed(False, 'on'))
        self.assertFalse(f.has_changed(True, 'on'))
        self.assertTrue(f.has_changed(True, ''))
        # Initial value may have mutated to a string due to show_hidden_initial (#19537)
        self.assertTrue(f.has_changed('False', 'on'))
        # HiddenInput widget sends string values for boolean but doesn't clean them in value_from_datadict
        self.assertFalse(f.has_changed(False, 'False'))
        self.assertFalse(f.has_changed(True, 'True'))
        self.assertTrue(f.has_changed(False, 'True'))
        self.assertTrue(f.has_changed(True, 'False'))

    def test_disabled_has_changed_gate(self):
        f = BooleanField(disabled=True)
        self.assertIs(f.has_changed('True', 'False'), False)

from django.test import TestCase, SimpleTestCase
from django.contrib.postgres.forms.array import SplitArrayField, SplitArrayWidget
from django.forms.widgets import CheckboxInput

class TestSplitArrayFieldWithBooleanField(SimpleTestCase):
    def test_checkbox_input_checked_state_repro(self):
        initial_data = [False, True, False]
        split_array_field = SplitArrayField(BooleanField(), size=len(initial_data))
        split_array_widget = SplitArrayWidget(widget=CheckboxInput(), size=len(initial_data))
        contexts = [split_array_widget.widget.get_context(name='test', value=value, attrs={})['widget'] for value in initial_data]
        self.assertNotIn('checked', contexts[0]['attrs'])
        self.assertIn('checked', contexts[1]['attrs'])
        self.assertNotIn('checked', contexts[2]['attrs'])
