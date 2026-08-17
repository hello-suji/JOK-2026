import os
import sys

sys.path.insert(0, os.path.abspath('.'))

extensions = ['sphinx.ext.autodoc']

# The suffix of source filenames.
source_suffix = '.rst'

autodoc_mock_imports = [
    'dummy'
]

nitpicky = True

import tempfile
import os
from sphinx.application import Sphinx
from sphinx.ext.autodoc import DocumenterBridge
from sphinx.ext.autodoc.directive import AutodocDirective
from sphinx.util.docstrings import prepare_docstring

def test_private_members_specific_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        confdir = outdir = doctreedir = tmpdir
        buildername = 'html'
        os.makedirs(srcdir)
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write('test module\n============\n.. automodule:: example_module\n   :private-members:')
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        app.build()
        assert 'private_function' in app.env.domains['py'].data['objects']
