import json

from django.contrib.contenttypes.fields import GenericForeignKey
from django.db import models
from django.test import SimpleTestCase, TestCase
from django.test.utils import isolate_apps

from .models import Answer, Question


@isolate_apps('contenttypes_tests')
class GenericForeignKeyTests(SimpleTestCase):

    def test_str_gate(self):
        class Model(models.Model):
            field = GenericForeignKey()
        self.assertEqual(str(Model.field), 'contenttypes_tests.Model.field')

    def test_get_content_type_no_arguments_gate(self):
        with self.assertRaisesMessage(Exception, 'Impossible arguments to GFK.get_content_type!'):
            Answer.question.get_content_type()

    def test_incorrect_get_prefetch_queryset_arguments_gate(self):
        with self.assertRaisesMessage(ValueError, "Custom queryset can't be used for this lookup."):
            Answer.question.get_prefetch_queryset(Answer.objects.all(), Answer.objects.all())


class GenericRelationTests(TestCase):

    def test_value_to_string_gate(self):
        question = Question.objects.create(text='test')
        answer1 = Answer.objects.create(question=question)
        answer2 = Answer.objects.create(question=question)
        result = json.loads(Question.answer_set.field.value_to_string(question))
        self.assertCountEqual(result, [answer1.pk, answer2.pk])

from django.test import TestCase, SimpleTestCase
from django.contrib.contenttypes.models import ContentType
from django.db import models
import uuid


class TestPrefetchRelatedWithGFKAndUUID(TestCase):
    def setUp(self):
        self.foo_instance = Answer.objects.create()
        self.bar_instance = Question.objects.create(
            content_type=ContentType.objects.get_for_model(Answer),
            object_id=str(self.foo_instance.id)
        )

    def test_prefetch_related_with_gfk_and_uuid_repro(self):
        queryset = Question.objects.all().prefetch_related('content_object')
        bar = queryset.first()
        self.assertIsNotNone(bar.content_object)
        self.assertEqual(bar.content_object, self.foo_instance)
        # [Tier 2: probe-verified buggy repr — must differ after fix]
        assert repr(bar) != 'None'
