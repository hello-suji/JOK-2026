from django import forms


class FormFieldAssertionsMixin:
    def assertWidgetRendersTo(self, field, to):
        class Form(forms.Form):
            f = field

        self.assertHTMLEqual(str(Form()["f"]), to)

from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.test import TestCase


class TestUserAdminPasswordLink(TestCase):
    def test_password_reset_link_with_to_field_repro(self):
        user = User.objects.create_user(username='testuser', password='testpass')
        user_admin = UserAdmin(User, None)
        form = user_admin.get_form(None, obj=user)()
        password_field = form.fields['password']
        self.assertNotIn('../password/', password_field.help_text)
        self.assertIn(f'../../{user.pk}/password/', password_field.help_text)
