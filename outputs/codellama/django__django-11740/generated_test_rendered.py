import uuid

from django.db import models
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
from django.db.migrations.state import ModelState, ProjectState


def test_uuidfield_to_foreignkey_creates_cross_app_dependency_repro():
    before = ProjectState()
    before.add_model(ModelState('testapp1', 'App1', [
        ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)),
        ('another_app', models.UUIDField(null=True, blank=True)),
    ]))
    before.add_model(ModelState('testapp2', 'App2', [
        ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)),
    ]))
    after = ProjectState()
    after.add_model(ModelState('testapp1', 'App1', [
        ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)),
        ('another_app', models.ForeignKey('testapp2.App2', null=True, blank=True, on_delete=models.SET_NULL)),
    ]))
    after.add_model(ModelState('testapp2', 'App2', [
        ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)),
    ]))
    changes = MigrationAutodetector(before, after, NonInteractiveMigrationQuestioner())._detect_changes()
    dependencies = changes['testapp1'][0].dependencies
    assert ('testapp2', '__first__') in dependencies
