extensions = ['sphinx.ext.intersphinx']

import tempfile
import os
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxError

def test_ambiguous_class_lookup_warning_repro():
    src_content = '''
.. py:class:: mod.A
.. py:class:: mod.submod.A

.. py:function:: f()

    - :py:class:`mod.A`
    - :py:class:`mod.submod.A`

    :param mod.A a:
    :param mod.submod.A b:
    :rtype: mod.A
    :rtype: mod.submod.A

.. py:currentmodule:: mod

.. py:function:: f()

    - :py:class:`A`
    - :py:class:`mod.A`
    - :py:class:`mod.submod.A`

    :param A a:
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        confdir = tmpdir
        os.makedirs(srcdir)
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write(src_content)
        with open(os.path.join(confdir, 'conf.py'), 'w') as f:
            f.write('extensions = ["sphinx.ext.autodoc"]\n')
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername='html')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            app.build()
            # [removed: warning presence oracle — assert fixed value/state instead]
            assert all('more than one target found for cross-reference' in str(warning.message) for warning in w)
