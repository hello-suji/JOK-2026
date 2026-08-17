extensions = ['sphinx.ext.intersphinx']

import tempfile
import os
from sphinx.application import Sphinx
from sphinx.builders.linkcheck import CheckExternalLinksBuilder

def test_linkcheck_reports_http_errors_instead_of_anchor_not_found_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'source')
        confdir = srcdir
        outdir = os.path.join(tmpdir, 'output')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        buildername = 'linkcheck'
        os.makedirs(srcdir)
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write('\n`foo <https://google.com/test.txt#test>`_\n')
        with open(os.path.join(srcdir, 'conf.py'), 'w') as f:
            f.write('extensions = []\nlinkcheck_anchors = True\n')
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername)
        app.build()
        results = app.env.linkcheck_results
        assert any('HTTP error' in msg for _, msg in results if 'https://google.com/test.txt#test' in msg)
