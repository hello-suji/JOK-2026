

from django.test import TestCase
from django.template.loader import render_to_string

class TestJsonScriptOptionalElementId(TestCase):
    def test_json_script_without_element_id_repro(self):
        context = {'data': {'key': 'value'}}
        rendered = render_to_string('template_with_json_script.html', context)
        self.assertTrue(rendered.strip().startswith('<script type="application/json">'))
        self.assertFalse(' id="' in rendered)
