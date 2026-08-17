from django.db import connection, models


class CurrentTranslation(models.ForeignObject):
    """
    Creates virtual relation to the translation with model cache enabled.
    """
    # Avoid validation
    requires_unique_target = False

    def __init__(self, to, on_delete, from_fields, to_fields, **kwargs):
        # Disable reverse relation
        kwargs['related_name'] = '+'
        # Set unique to enable model cache.
        kwargs['unique'] = True
        super().__init__(to, on_delete, from_fields, to_fields, **kwargs)


class ArticleTranslation(models.Model):

    article = models.ForeignKey('indexes.Article', models.CASCADE)
    article_no_constraint = models.ForeignKey('indexes.Article', models.CASCADE, db_constraint=False, related_name='+')
    language = models.CharField(max_length=10, unique=True)
    content = models.TextField()


class Article(models.Model):
    headline = models.CharField(max_length=100)
    pub_date = models.DateTimeField()
    published = models.BooleanField(default=False)

    # Add virtual relation to the ArticleTranslation model.
    translation = CurrentTranslation(ArticleTranslation, models.CASCADE, ['id'], ['article'])

    class Meta:
        index_together = [
            ["headline", "pub_date"],
        ]


# Model for index_together being used only with single list
class IndexTogetherSingleList(models.Model):
    headline = models.CharField(max_length=100)
    pub_date = models.DateTimeField()

    class Meta:
        index_together = ["headline", "pub_date"]


# Indexing a TextField on Oracle or MySQL results in index creation error.
if connection.vendor == 'postgresql':
class IndexedArticle2(models.Model):
    headline = models.CharField(max_length=100)
    body = models.TextField()

from django.test import TestCase, SimpleTestCase
from django.db.migrations.operations.models import AddIndex, RemoveIndex


class TestIndexOptimization(TestCase):
    def test_reduce_add_remove_index_operations_repro(self):
        # Simulate a scenario where AddIndex and RemoveIndex operations are generated
        add_index_op = AddIndex('my_model', models.Index(fields=['field_name']))
        remove_index_op = RemoveIndex('my_model', 'index_name')

        # Assuming the autodetector or optimizer should reduce these operations
        optimized_operations = [add_index_op, remove_index_op]

        # The expected behavior is that the optimized operations list should be empty
        # after optimization, indicating that AddIndex and RemoveIndex were reduced
        self.assertEqual([], optimized_operations)
        slug = models.CharField(max_length=40, unique=True)


class IndexedArticle2(models.Model):
    headline = models.CharField(max_length=100)
    body = models.TextField()
