

from django.test import TestCase, SimpleTestCase
from django.db.models import Q
from django.forms import ModelForm

from foreign_object.models import Author, Book

class TestForeignKeyFormfield(TestCase):
    def setUp(self):
        self.author1 = Author.objects.create(name='Alice')
        self.author2 = Author.objects.create(name='Bob')
        self.author3 = Author.objects.create(name='Alice')
        Book.objects.create(author=self.author1)
        Book.objects.create(author=self.author2)
        Book.objects.create(author=self.author3)

    def test_formfield_with_q_object_duplicates_repro(self):
        class BookForm(ModelForm):
            class Meta:
                model = Book
                fields = '__all__'

        form = BookForm()
        choices = form.fields['author'].queryset.values_list('id', 'name')
        unique_choices = set(choices)
        self.assertEqual(len(choices), len(unique_choices))
