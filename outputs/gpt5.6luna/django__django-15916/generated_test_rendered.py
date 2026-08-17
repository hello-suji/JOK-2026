from django import forms


class FormFieldAssertionsMixin:
    def assertWidgetRendersTo(self, field, to):
        class Form(forms.Form):
            f = field

        self.assertHTMLEqual(str(Form()["f"]), to)

from django.forms.models import modelform_factory
from django.contrib.auth.models import User

def all_required(field, **kwargs):
    formfield = field.formfield(**kwargs)
    formfield.required = True
    return formfield

class MyForm(forms.ModelForm):
    formfield_callback = all_required

    class Meta:
        model = User
        fields = '__all__'

class FactoryForm(MyForm):
    pass

def test_modelform_factory_with_callback_repro():
    FormClass = modelform_factory(User, form=FactoryForm)
    form = FormClass()
    for field in form.fields.values():
        assert field.required
