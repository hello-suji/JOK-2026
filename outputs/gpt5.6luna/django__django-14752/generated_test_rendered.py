"""
A custom AdminSite for AdminViewPermissionsTest.test_login_has_permission().
"""
from django.contrib import admin
from django.contrib.auth import get_permission_codename
from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError

from . import admin as base_admin, models

PERMISSION_NAME = 'admin_views.%s' % get_permission_codename('change', models.Article._meta)


class PermissionAdminAuthenticationForm(AuthenticationForm):
    def confirm_login_allowed(self, user):
        if not user.is_active or not (user.is_staff or user.has_perm(PERMISSION_NAME)):
            raise ValidationError('permission denied')


class HasPermissionAdmin(admin.AdminSite):
    login_form = PermissionAdminAuthenticationForm

    def has_permission(self, request):
        return (
            request.user.is_active and
            (request.user.is_staff or request.user.has_perm(PERMISSION_NAME))
        )


site = HasPermissionAdmin(name="has_permission_admin")
site.register(models.Article, base_admin.ArticleAdmin)

from django.contrib.admin.views.autocomplete import AutocompleteJsonView
from django.http import HttpRequest
from django.test import TestCase
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import Permission


class TestAutocompleteJsonView(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.user.is_staff = True
        self.user.save()
        content_type = ContentType.objects.get_for_model(User)
        permission = Permission.objects.create(codename='view_user', name='Can view user', content_type=content_type)
        self.user.user_permissions.add(permission)
        self.client.login(username='testuser', password='testpass')

    def test_custom_extra_fields_in_autocomplete_response_repro(self):
        request = HttpRequest()
        request.GET = {'term': 'test'}
        request.user = self.user
        view = AutocompleteJsonView()
        view.model_admin = admin.ModelAdmin(User, admin.site)
        view.queryset = User.objects.all()
        response = view.get(request)
        self.assertIn('notes', response.json()['results'][0])
