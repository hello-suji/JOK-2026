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

from __future__ import annotations

from typing import Any, Dict
from sphinx.application import Sphinx
from sphinx.ext.autodoc import DocumenterBridge
from sphinx.ext.autodoc.directive import AutodocDirective
import tempfile
import os

JSONObject = Dict[str, Any]

def sphinx_doc(data: JSONObject) -> JSONObject:
    """Does it work.

    Args:
        data: Does it args.

    Returns:
        Does it work in return.
    """
    return {}

def test_autodoc_type_aliases_with_description_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        confdir = srcdir
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctree')
        buildername = 'html'
        os.makedirs(srcdir)
        with open(os.path.join(srcdir, 'conf.py'), 'w') as f:
            f.write("""
autodoc_typehints = 'description'
autodoc_type_aliases = {
    'JSONObject': 'types.JSONObject',
}
""")
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write("""
.. autofunction:: sphinx_doc
""")
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        app.build()
        with open(os.path.join(outdir, 'index.html'), 'r') as f:
            content = f.read()
        assert 'Dict[str, Any]' in content
        assert 'types.JSONObject' not in content
