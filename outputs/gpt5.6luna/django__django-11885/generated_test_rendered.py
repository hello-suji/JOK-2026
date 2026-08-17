"""
Adding hooks before/after saving and deleting

To execute arbitrary code around ``save()`` and ``delete()``, just subclass
the methods.
"""
from django.db import models


class Person(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []

    def __str__(self):
        return "%s %s" % (self.first_name, self.last_name)

    def save(self, *args, **kwargs):
        self.data.append("Before save")
        # Call the "real" save() method
        super().save(*args, **kwargs)
        self.data.append("After save")

    def delete(self):
        self.data.append("Before deletion")
        # Call the "real" delete() method
        super().delete()
        self.data.append("After deletion")

from django.db import models
from django.db.models.deletion import Collector
from django.contrib.auth.models import User, Group

class TestDeleteCascadeCombination(models.TestCase):
    def setUp(self):
        self.user1 = User.objects.create(username='user1')
        self.user2 = User.objects.create(username='user2')
        self.group1 = Group.objects.create(name='group1')
        self.group2 = Group.objects.create(name='group2')
        self.group1.user_set.add(self.user1)
        self.group2.user_set.add(self.user2)
        self.group1.user_set.add(self.user2)

    def test_delete_cascade_combination_repro(self):
        collector = Collector(using='default')
        collector.collect([self.group1, self.group2])
        sqls = collector.sqls
        self.assertEqual(len(sqls), 1)
        self.assertIn('DELETE FROM auth_user_groups', sqls[0][0])
        self.assertIn('WHERE auth_user_groups.group_id IN', sqls[0][0])
