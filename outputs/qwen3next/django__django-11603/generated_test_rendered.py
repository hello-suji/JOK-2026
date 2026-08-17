from django.test import SimpleTestCase
from django.utils import translation

from ...utils import setup


class I18nFiltersTests(SimpleTestCase):
    libraries = {
        'custom': 'template_tests.templatetags.custom',
        'i18n': 'django.templatetags.i18n',
    }

    @setup({'i18n32': '{% load i18n %}{{ "hu"|language_name }} '
                      '{{ "hu"|language_name_local }} {{ "hu"|language_bidi }} '
                      '{{ "hu"|language_name_translated }}'})
    def test_i18n32_gate(self):
        output = self.engine.render_to_string('i18n32')
        self.assertEqual(output, 'Hungarian Magyar False Hungarian')

        with translation.override('cs'):
            output = self.engine.render_to_string('i18n32')
            self.assertEqual(output, 'Hungarian Magyar False maďarsky')

    @setup({'i18n33': '{% load i18n %}'
                      '{{ langcode|language_name }} {{ langcode|language_name_local }} '
                      '{{ langcode|language_bidi }} {{ langcode|language_name_translated }}'})
    def test_i18n33_gate(self):
        output = self.engine.render_to_string('i18n33', {'langcode': 'nl'})
        self.assertEqual(output, 'Dutch Nederlands False Dutch')

        with translation.override('cs'):
            output = self.engine.render_to_string('i18n33', {'langcode': 'nl'})
            self.assertEqual(output, 'Dutch Nederlands False nizozemsky')

    @setup({'i18n38_2': '{% load i18n custom %}'
                        '{% get_language_info_list for langcodes|noop:"x y" as langs %}'
                        '{% for l in langs %}{{ l.code }}: {{ l.name }}/'
                        '{{ l.name_local }}/{{ l.name_translated }} '
                        'bidi={{ l.bidi }}; {% endfor %}'})
    def test_i18n38_2_gate(self):
        with translation.override('cs'):
            output = self.engine.render_to_string('i18n38_2', {'langcodes': ['it', 'fr']})
        self.assertEqual(
            output,
            'it: Italian/italiano/italsky bidi=False; '
            'fr: French/français/francouzsky bidi=False; '
        )

from django.test import TestCase, SimpleTestCase
from django.db.models import Avg, Sum
from django.contrib.auth.models import User

class TestAggregationSupport(TestCase):
    def setUp(self):
        User.objects.create(username='user1', first_name='John', last_name='Doe')
        User.objects.create(username='user2', first_name='Jane', last_name='Doe')
        User.objects.create(username='user3', first_name='John', last_name='Smith')

    def test_avg_with_distinct_repro(self):
        avg_first_name_length = User.objects.aggregate(avg_length=Avg('first_name__length', distinct=True))['avg_length']
        self.assertIsNotNone(avg_first_name_length)
