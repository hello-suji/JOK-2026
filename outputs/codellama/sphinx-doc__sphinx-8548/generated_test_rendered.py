# example.sphinx


class DummyClass(object):
    pass

import unittest
import tempfile
import os
from sphinx.application import Sphinx
from sphinx.ext.autodoc import Documenter

class BaseClass:
    def __init__(self):
        self.base_attribute = 'base_value'

class DerivedClass(BaseClass):
    pass

class TestInheritedMembers(unittest.TestCase):
    def test_inherited_members_documentation_repro(self):
        srcdir = tempfile.mkdtemp()
        confdir = srcdir
        outdir = tempfile.mkdtemp()
        doctreedir = tempfile.mkdtemp()
        buildername = 'html'

        with open(os.path.join(srcdir, 'conf.py'), 'w') as f:
            f.write("extensions = ['sphinx.ext.autodoc']")
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write("""
.. autoclass:: DerivedClass
   :members:
   :inherited-members:
""")

        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        app.build(force_all=True)

        # Check if the base attribute is documented
        with open(os.path.join(outdir, 'index.html'), 'r') as f:
            html_content = f.read()
        self.assertIn('base_attribute', html_content)
