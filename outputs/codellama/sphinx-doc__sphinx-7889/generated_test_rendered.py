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
from sphinx.ext.autodoc.mock import _make_subclass
from typing import TypeVar

def test_make_subclass_with_generic_fails_pre_patch_repro():
    T = TypeVar('T')
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        confdir = srcdir
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctree')
        buildername = 'html'
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        try:
            _make_subclass(app, 'GenericClass', (T,), {})
            assert False, "Expected TypeError was not raised"
        except TypeError:
            pass
