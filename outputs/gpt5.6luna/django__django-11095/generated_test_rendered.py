

from django.contrib import admin
from django.test import TestCase
from django.contrib.auth.models import User
from tests.admin_inlines.models import Author, Book

class TestCustomModelAdmin(TestCase):
    def test_get_inline_instances_dynamic_behavior_repro(self):
        class CustomModelAdmin(admin.ModelAdmin):
            def get_inlines(self, request, obj=None):
                if obj and obj.name == 'John Doe':
                    return [BookInline]
                return []

        class BookInline(admin.TabularInline):
            model = Book

        self.client.force_login(User.objects.create_superuser('superuser', 'email@example.com', 'password'))
        request = self.client.get('/admin/admin_inlines/author/').wsgi_request
        obj = Author.objects.create(name='John Doe')
        inline_instances = CustomModelAdmin(Book).get_inline_instances(request, obj)
        self.assertEqual(len(inline_instances), 1)
