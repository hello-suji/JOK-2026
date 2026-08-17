

from django.test import TestCase
from django.utils.formats import get_format
from django.utils.translation import gettext_lazy as _

class TestGetFormatWithLazyString(TestCase):
    def test_get_format_with_lazy_string_repro(self):
        lazy_string = _('Y-m-d')
        result = get_format('DATE_FORMAT', lazy_string)
        self.assertIsNotNone(result)
        # [Tier 2: probe-verified buggy repr — must differ after fix]
        assert repr(result) != 'TypeError: getattr(): attribute name must be string'
