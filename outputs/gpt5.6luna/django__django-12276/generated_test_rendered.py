from urllib.parse import urljoin

from django.contrib.staticfiles import storage
from django.forms import Media
from django.templatetags.static import static
from django.test import SimpleTestCase, override_settings


class StaticTestStorage(storage.StaticFilesStorage):
    def url(self, name):
        return urljoin('https://example.com/assets/', name)


@override_settings(
    STATIC_URL='http://media.example.com/static/',
    INSTALLED_APPS=('django.contrib.staticfiles',),
    STATICFILES_STORAGE='staticfiles_tests.test_forms.StaticTestStorage',
)
class StaticFilesFormsMediaTestCase(SimpleTestCase):
    def test_absolute_url(self):
        m = Media(
            css={'all': ('path/to/css1', '/path/to/css2')},
            js=(
                '/path/to/js1',
                'http://media.other.com/path/to/js2',
                'https://secure.other.com/path/to/js3',
                static('relative/path/to/js4'),
            ),
        )
        self.assertEqual(
            str(m),
            """<link href="https://example.com/assets/path/to/css1" type="text/css" media="all" rel="stylesheet">
<link href="/path/to/css2" type="text/css" media="all" rel="stylesheet">
<script src="/path/to/js1"></script>
<script src="http://media.other.com/path/to/js2"></script>
<script src="https://secure.other.com/path/to/js3"></script>
<script src="https://example.com/assets/relative/path/to/js4"></script>"""
        )

from django.test import TestCase, SimpleTestCase
from django.forms.widgets import FileInput

class TestFileInputRequiredAttribute(SimpleTestCase):
    def test_use_required_attribute_with_initial_data_repro(self):
        widget = FileInput()
        self.assertFalse(widget.use_required_attribute(initial='somefile.txt'))
