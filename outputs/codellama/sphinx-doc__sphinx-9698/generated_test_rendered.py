exclude_patterns = ['_build']
extensions = [
	'sphinx.ext.intersphinx',
]

import tempfile
import os
from sphinx.application import Sphinx
from sphinx.ext.autodoc import DocumenterBridge
from sphinx.domains.python import PyMethodDocumenter

def test_index_entry_without_parens_repro():
    src_content = '''
.. py:method:: Foo.bar
   :property:

.. py:property:: Foo.baz
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        confdir = tmpdir
        os.makedirs(srcdir)
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write(src_content)
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername='text')
        app.build()
        env = app.env
        index_entries = env.domaindata['py']['objects']
        assert ('bar', 'Foo.bar', 'method', '', '', '', '') in index_entries
        assert ('baz', 'Foo.baz', 'property', '', '', '', '') in index_entries
        assert ('bar', 'Foo.bar()', 'method', '', '', '', '') not in index_entries
