from unittest.mock import patch

from django.core.mail import EmailMessage
from django.test import SimpleTestCase

class EmailUnicodeDNSRegressionTests(SimpleTestCase):
    @patch('django.core.mail.message.DNS_NAME', '漢字')
    def test_message_id_uses_punycode_for_non_ascii_dns_name_repro(self):
        email = EmailMessage('subject', '', 'from@example.com', ['to@example.com'])
        email.encoding = 'iso-8859-1'
        message = email.message()
        self.assertIn('xn--p8s937b', message['Message-ID'])
