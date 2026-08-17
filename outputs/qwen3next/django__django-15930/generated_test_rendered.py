from django import forms


class FormFieldAssertionsMixin:
    def assertWidgetRendersTo(self, field, to):
        class Form(forms.Form):
            f = field

        self.assertHTMLEqual(str(Form()["f"]), to)

from django.test import TestCase, SimpleTestCase
from django.db.models import Case, When, Value, BooleanField
from django.db.models.query import QuerySet
from django.contrib.auth.models import User

class TestAnnotateWithEmptyQ(TestCase):
    def test_annotate_with_empty_q_repro(self):
        User.objects.create(username='testuser1')
        User.objects.create(username='testuser2')
        annotated_users = User.objects.annotate(
            _a=Case(
                When(~Q(pk__in=[]), then=Value(True)),
                default=Value(False),
                output_field=BooleanField(),
            )
        ).order_by('-_a').values('pk')
        self.assertIsInstance(annotated_users, QuerySet)
        self.assertTrue(all(user['_a'] for user in annotated_users))
