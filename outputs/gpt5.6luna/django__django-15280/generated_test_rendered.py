

from django.test import TestCase
from django.db.models import CharField, EmailField, OneToOneField
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType


class TestDeferredFieldsPrefetch(TestCase):
    def test_only_related_queryset_repro(self):
        user = User.objects.create(email='test@example.com', kind='ADMIN')
        profile = user.profile_set.create(full_name='John Doe')
        queryset = User.objects.only('email').prefetch_related('profile_set')
        with self.assertNumQueries(0):
            user = queryset.first()
            self.assertEqual(user.profile_set.first().user.get_deferred_fields(), {'kind'})
