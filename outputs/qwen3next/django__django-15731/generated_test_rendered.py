from django import forms


class FormFieldAssertionsMixin:
    def assertWidgetRendersTo(self, field, to):
        class Form(forms.Form):
            f = field

        self.assertHTMLEqual(str(Form()["f"]), to)

from django.test import TestCase, SimpleTestCase
import inspect
from django.forms.fields import CharField

class TestCharFieldSignature(TestCase):
    def test_charfield_init_signature_repro(self):
        sig = inspect.signature(CharField.__init__)
        self.assertEqual(list(sig.parameters.keys()), ['self', '*args', '**kwargs'])
