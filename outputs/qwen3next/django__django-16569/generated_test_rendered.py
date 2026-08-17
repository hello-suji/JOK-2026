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
            widget = Select(attrs={"class": "my-custom-class"})

        class TestForm(Form):
            field1 = CustomChoiceField(choices=[])
            field2 = CustomChoiceField(choices=[])

        f = TestForm()
        f.fields["field1"].choices = [("1", "1")]
        f.fields["field2"].choices = [("2", "2")]
        self.assertEqual(f.fields["field1"].widget.choices, [("1", "1")])
        self.assertEqual(f.fields["field2"].widget.choices, [("2", "2")])


class DisabledFieldTests(SimpleTestCase):
    def test_disabled_field_has_changed_always_false_gate(self):
        disabled_field = Field(disabled=True)
        self.assertFalse(disabled_field.has_changed("x", "y"))

from django.forms import CharField, Form
from django.forms.formsets import formset_factory
from django.test import TestCase

class TestFormSetAddFields(TestCase):
    def test_add_fields_with_none_index_repro(self):
        class MyForm(Form):
            field1 = CharField()

        MyFormSet = formset_factory(MyForm, can_delete=True, can_delete_extra=False)
        formset = MyFormSet()
        empty_form = formset.empty_form
        # This should not raise a TypeError
        empty_form.add_fields(empty_form, 0)
        self.assertEqual(len(empty_form.fields), 2)  # Assuming the default management form fields are added
