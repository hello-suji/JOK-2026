extensions = ['sphinx.ext.intersphinx']

import tempfile
import os
from sphinx.application import Sphinx
from sphinx.ext.autodoc import DocumenterBridge

def test_empty_all_attribute_ignored_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        confdir = srcdir
        os.makedirs(srcdir)
        example_content = """__all__ = []

def foo():
    \"docstring\"

def bar():
    \"docstring\"

def baz():
    \"docstring\""""
        index_rst_content = """.. automodule:: example
   :members:"""
        with open(os.path.join(srcdir, 'example.py'), 'w') as f:
            f.write(example_content)
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write(index_rst_content)
        with open(os.path.join(confdir, 'conf.py'), 'w') as f:
            f.write('extensions = [\"sphinx.ext.autodoc\"]\n')
        app = Sphinx(srcdir, confdir, outdir, doctreedir, 'text')
        app.build()
        output_path = os.path.join(outdir, 'index.txt')
        with open(output_path, 'r') as f:
            output = f.read()
        assert 'foo' not in output
        assert 'bar' not in output
        assert 'baz' not in output
