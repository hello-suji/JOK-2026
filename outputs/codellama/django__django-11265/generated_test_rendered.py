from django.core.exceptions import FieldError
from django.db.models import FilteredRelation, Q
from django.test import TestCase

from .models import Author, Book

class FilteredRelationExcludeRegressionTests(TestCase):
    def test_exclude_on_filtered_relation_join_does_not_raise_field_error_repro(self):
        author = Author.objects.create(name='Alice')
        Book.objects.create(title='poem by alice', author=author)
        queryset = Author.objects.annotate(
            book_alice=FilteredRelation(
                'book', condition=Q(book__title__iexact='poem by alice')
            )
        )
        try:
            list(queryset.exclude(book_alice__isnull=False))
        except FieldError as exc:
            self.fail('FilteredRelation exclude raised FieldError: %s' % exc)
