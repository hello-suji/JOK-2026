from django.forms import ChoiceField, Field, Form, Select
from django.test import SimpleTestCase


class BasicFieldsTests(SimpleTestCase):

    def test_field_sets_widget_is_required_gate(self):
        self.assertTrue(Field(required=True).widget.is_required)
        self.assertFalse(Field(required=False).widget.is_required)

    def test_cooperative_multiple_inheritance_gate(self):
        class A:
            def __init__(self):
                self.class_a_var = True
                super().__init__()

        class ComplexField(Field, A):
            def __init__(self):
                super().__init__()

        f = ComplexField()
        self.assertTrue(f.class_a_var)

    def test_field_deepcopies_widget_instance_gate(self):
        class CustomChoiceField(ChoiceField):
            widget = Select(attrs={'class': 'my-custom-class'})

        class TestForm(Form):
            field1 = CustomChoiceField(choices=[])
            field2 = CustomChoiceField(choices=[])

        f = TestForm()
        f.fields['field1'].choices = [('1', '1')]
        f.fields['field2'].choices = [('2', '2')]
        self.assertEqual(f.fields['field1'].widget.choices, [('1', '1')])
        self.assertEqual(f.fields['field2'].widget.choices, [('2', '2')])


class DisabledFieldTests(SimpleTestCase):
    def test_disabled_field_has_changed_always_false_gate(self):
        disabled_field = Field(disabled=True)
        self.assertFalse(disabled_field.has_changed('x', 'y'))

from django.test import TestCase
from django.db.models import Model, BooleanField, AutoField

class SaveTestCase(TestCase):
    def setUp(self):
        class Item(Model):
            id = AutoField(primary_key=True)
            is_active = BooleanField(default=True)
        self.Item = Item
        self.item = self.Item.objects.create(is_active=True)

    def test_f_true_repro(self):
        self.item.is_active = False
        self.item.save()
        self.item.refresh_from_db()
        self.assertTrue(not self.item.is_active)
