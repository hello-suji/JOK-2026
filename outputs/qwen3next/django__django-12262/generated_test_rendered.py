import json

from django.template.loader import render_to_string
from django.test import SimpleTestCase


class TestTemplates(SimpleTestCase):
    def test_javascript_escaping_gate(self):
        context = {
            'inline_admin_formset': {
                'inline_formset_data': json.dumps({
                    'formset': {'prefix': 'my-prefix'},
                    'opts': {'verbose_name': 'verbose name\\'},
                }),
            },
        }
        output = render_to_string('admin/edit_inline/stacked.html', context)
        self.assertIn('&quot;prefix&quot;: &quot;my-prefix&quot;', output)
        self.assertIn('&quot;verbose_name&quot;: &quot;verbose name\\\\&quot;', output)

        output = render_to_string('admin/edit_inline/tabular.html', context)
        self.assertIn('&quot;prefix&quot;: &quot;my-prefix&quot;', output)
        self.assertIn('&quot;verbose_name&quot;: &quot;verbose name\\\\&quot;', output)

from django.test import TestCase, SimpleTestCase
class TestCustomTemplateTags(SimpleTestCase):
    def test_hello_with_keyword_argument_repro(self):
        template_content = '{% load asgi_urls %}{% hello greeting="hi" %}'
        rendered_content = render_to_string(template_content)
        self.assertEqual(rendered_content.strip(), 'hi')
