exclude_patterns = ['_build']
extensions = [
	'sphinx.ext.intersphinx',
]

import tempfile
import os
from sphinx.application import Sphinx

def test_classmethod_property_documentation_repro():
    class A:
        @classmethod
        @property
        def f(cls):
            """Some class property."""
            return "property"

    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        confdir = os.path.join(tmpdir, 'conf')
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        buildername = 'html'

        os.makedirs(srcdir)
        os.makedirs(confdir)

        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write('A.f\n====\n\n.. autofunction:: A.f\n')

        with open(os.path.join(confdir, 'conf.py'), 'w') as f:
            f.write('extensions = ["sphinx.ext.autodoc"]\n')

        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        app.build()

        with open(os.path.join(outdir, 'index.html'), 'r') as f:
            content = f.read()
            assert 'Some class property.' in content
