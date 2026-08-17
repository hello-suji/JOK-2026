from django.contrib.admin.tests import AdminSeleniumTestCase
from django.test import override_settings
from django.urls import reverse

from ..models import Article


@override_settings(ROOT_URLCONF='forms_tests.urls')
class LiveWidgetTests(AdminSeleniumTestCase):

    available_apps = ['forms_tests'] + AdminSeleniumTestCase.available_apps

    def test_textarea_trailing_newlines_gate(self):
        """
        A roundtrip on a ModelForm doesn't alter the TextField value
        """
        from selenium.webdriver.common.by import By
        article = Article.objects.create(content="\nTst\n")
        self.selenium.find_element_by_id('submit').click()
        article = Article.objects.get(pk=article.pk)
        self.assertEqual(article.content, "\r\nTst\r\n")

from django.test import TestCase, SimpleTestCase
from django.forms.boundfield import BoundField
from django.forms.widgets import CheckboxSelectMultiple


class TestCheckboxSelectMultipleSubwidgets(TestCase):
    def test_subwidgets_id_assignment_repro(self):
        widget = CheckboxSelectMultiple()
        field = widget.field = type('MockField', (), {'widget': widget})()
        bound_field = BoundField(field, [], None)
        attrs = {'id': 'custom_id'}
        subwidgets = bound_field.subwidgets('html_name', [], attrs=attrs)
        self.assertEqual(subwidgets[0].data['attrs']['id'], 'custom_id')
        self.assertEqual(article.content, "\r\nTst\r\n")
