from django.urls import include, path, register_converter

from . import converters, views

register_converter(converters.Base64Converter, 'base64')

subsubpatterns = [
    path('<base64:last_value>/', views.empty_view, name='subsubpattern-base64'),
]

subpatterns = [
    path('<base64:value>/', views.empty_view, name='subpattern-base64'),
    path(
        '<base64:value>/',
        include((subsubpatterns, 'second-layer-namespaced-base64'), 'instance-ns-base64')
    ),
]

urlpatterns = [
    path('base64/<base64:value>/', views.empty_view, name='base64'),
    path('base64/<base64:base>/subpatterns/', include(subpatterns)),
    path('base64/<base64:base>/namespaced/', include((subpatterns, 'namespaced-base64'))),
]

from django.conf.urls.i18n import i18n_patterns
from django.http.response import HttpResponse
from django.test import TestCase
from django.urls import path

def bangiah(request):
    return HttpResponse('U!')

urlpatterns = i18n_patterns(
    path('', bangiah, name='bangiah'),
)

class TestBangiahWithLocale(TestCase):
    def test_bangiah_with_en_latn_us_repro(self):
        response = self.client.get('/en-latn-us/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), 'U!')
