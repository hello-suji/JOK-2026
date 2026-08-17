from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Test basic commands'
    requires_system_checks = False

    def add_arguments(self, parser):
        parser.add_argument('args', nargs='*')
        parser.add_argument('--option_a', '-a', default='1')
        parser.add_argument('--option_b', '-b', default='2')
        parser.add_argument('--option_c', '-c', default='3')

    def handle(self, *labels, **options):
        print('EXECUTE:BaseCommand labels=%s, options=%s' % (labels, sorted(options.items())))

from django.test import TestCase, SimpleTestCase
import re
from django.core.management import call_command
from django.core.management.base import CommandError

class TestMutuallyExclusiveGroupCommand(TestCase):
    def test_call_command_with_valid_kwargs_repro(self):
        try:
            result = call_command('my_command', shop_id=1)
            self.assertIsNone(result)  # Assuming the command returns None on success
        except CommandError:
            self.fail('call_command raised CommandError unexpectedly')
