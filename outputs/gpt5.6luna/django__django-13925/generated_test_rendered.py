default_app_config = 'apps.explicit_default_config_app.apps.ExplicitDefaultConfig'

import re
from django.test import TestCase
from django.core.management.validation import get_validation_errors
from io import StringIO
from django.contrib.auth.models import AbstractUser
from django.db import models

class TestInheritedPrimaryKey(TestCase):
    def test_inherited_primary_key_repro(self):
        class Entity(AbstractUser):
            pass

        class User(Entity):
            pass

        # Redirect stdout to capture validation errors
        output = StringIO()
        get_validation_errors(output)
        validation_output = output.getvalue()

        # Pre-patch, this should contain W042 warnings
        # Post-patch, this should not contain W042 warnings
        self.assertNotIn('W042', validation_output)
