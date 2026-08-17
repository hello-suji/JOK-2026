from django.core.exceptions import ImproperlyConfigured
from django.template import engines
from django.test import SimpleTestCase, override_settings


class TemplateUtilsTests(SimpleTestCase):

    @override_settings(TEMPLATES=[{'BACKEND': 'raise.import.error'}])
    def test_backend_import_error_gate(self):
        """
        Failing to import a backend keeps raising the original import error
        (#24265).
        """
        with self.assertRaisesMessage(ImportError, "No module named 'raise"):
            engines.all()
        with self.assertRaisesMessage(ImportError, "No module named 'raise"):
            engines.all()

    @override_settings(TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # Incorrect: APP_DIRS and loaders are mutually incompatible.
        'APP_DIRS': True,
        'OPTIONS': {'loaders': []},
    }])
    def test_backend_improperly_configured_gate(self):
        """
        Failing to initialize a backend keeps raising the original exception
        (#24265).
        """
        msg = 'app_dirs must not be set when loaders is defined.'
        with self.assertRaisesMessage(ImproperlyConfigured, msg):
            engines.all()
        with self.assertRaisesMessage(ImproperlyConfigured, msg):
            engines.all()

    @override_settings(TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
    }, {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
    }])
    def test_backend_names_must_be_unique_gate(self):
        msg = (
            "Template engine aliases aren't unique, duplicates: django. Set "
            "a unique NAME for each engine in settings.TEMPLATES."
        )
        with self.assertRaisesMessage(ImproperlyConfigured, msg):
            engines.all()

from django.test import TestCase, SimpleTestCase
from django.utils.html import escape

class TestEscapeFunction(SimpleTestCase):
    def test_escape_behavior_repro(self):
        input_string = '<script>alert("XSS")</script>'
        expected_output = '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'
        self.assertEqual(expected_output, escape(input_string))
