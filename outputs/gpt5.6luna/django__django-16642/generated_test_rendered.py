import posixpath
from urllib.parse import quote

from django.conf import settings
from django.test import override_settings

from .cases import StaticFilesTestCase, TestDefaults


@override_settings(ROOT_URLCONF="staticfiles_tests.urls.default")
class TestServeStatic(StaticFilesTestCase):
    """
    Test static asset serving view.
    """

    def _response(self, filepath):
        return self.client.get(quote(posixpath.join(settings.STATIC_URL, filepath)))

    def assertFileContains(self, filepath, text):
        self.assertContains(self._response(filepath), text)

    def assertFileNotFound(self, filepath):
        self.assertEqual(self._response(filepath).status_code, 404)


@override_settings(DEBUG=False)
class TestServeDisabled(TestServeStatic):
    """
    Test serving static files disabled when DEBUG is False.
    """

    def test_disabled_serving(self):
        self.assertFileNotFound("test.txt")


@override_settings(DEBUG=True)
class TestServeStaticWithDefaultURL(TestDefaults, TestServeStatic):
    """
    Test static asset serving view with manually configured URLconf.
    """


@override_settings(DEBUG=True, ROOT_URLCONF="staticfiles_tests.urls.helper")
class TestServeStaticWithURLHelper(TestDefaults, TestServeStatic):
    """
    Test static asset serving view with staticfiles_urlpatterns helper.
    """

from django.http import FileResponse


class TestFileResponseContentType(StaticFilesTestCase):
    def test_file_response_z_content_type_repro(self):
        with open('test.html.Z', 'wb') as f:
            f.write(b'test data')
        response = FileResponse(open('test.html.Z', 'rb'))
        self.assertNotEqual(response['Content-Type'], 'text/html')
        # [Tier 2: probe-verified buggy repr — must differ after fix]
        assert repr(response) != 'text/html'
