"""
    test_build_gettext
    ~~~~~~~~~~~~~~~~~~

    Test the build process with gettext builder with the test root.

    :copyright: Copyright 2007-2020 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import gettext
import os
import re
import subprocess
from subprocess import PIPE, CalledProcessError

import pytest

from sphinx.util.osutil import cd


@pytest.mark.sphinx('gettext', srcdir='root-gettext')
def test_build_gettext(app):
    # Generic build; should fail only when the builder is horribly broken.
    app.builder.build_all()

    # Do messages end up in the correct location?
    # top-level documents end up in a message catalog
    assert (app.outdir / 'extapi.pot').isfile()
    # directory items are grouped into sections
    assert (app.outdir / 'subdir.pot').isfile()

    # regression test for issue #960
    catalog = (app.outdir / 'markup.pot').read_text()
    assert 'msgid "something, something else, something more"' in catalog


@pytest.mark.sphinx('gettext', srcdir='root-gettext')
def test_msgfmt(app):
    app.builder.build_all()
    (app.outdir / 'en' / 'LC_MESSAGES').makedirs()
    with cd(app.outdir):
        try:
            args = ['msginit', '--no-translator', '-i', 'markup.pot', '--locale', 'en_US']
            subprocess.run(args, stdout=PIPE, stderr=PIPE, check=True)
        except OSError:
            pytest.skip()  # most likely msginit was not found
        except CalledProcessError as exc:
            print(exc.stdout)
            print(exc.stderr)
            assert False, 'msginit exited with return code %s' % exc.returncode

        assert (app.outdir / 'en_US.po').isfile(), 'msginit failed'
        try:
            args = ['msgfmt', 'en_US.po',
                    '-o', os.path.join('en', 'LC_MESSAGES', 'test_root.mo')]
            subprocess.run(args, stdout=PIPE, stderr=PIPE, check=True)
        except OSError:
            pytest.skip()  # most likely msgfmt was not found
        except CalledProcessError as exc:
            print(exc.stdout)
            print(exc.stderr)
            assert False, 'msgfmt exited with return code %s' % exc.returncode

        mo = app.outdir / 'en' / 'LC_MESSAGES' / 'test_root.mo'
        assert mo.isfile(), 'msgfmt failed'

    _ = gettext.translation('test_root', app.outdir, languages=['en']).gettext
    assert _("Testing various markup") == "Testing various markup"


@pytest.mark.sphinx(
    'gettext', testroot='intl', srcdir='gettext',
    confoverrides={'gettext_compact': False})
def test_gettext_index_entries(app):
    # regression test for #976
    app.builder.build(['index_entries'])

    _msgid_getter = re.compile(r'msgid "(.*)"').search

    def msgid_getter(msgid):
        m = _msgid_getter(msgid)
        if m:
            return m.groups()[0]
        return None

    pot = (app.outdir / 'index_entries.pot').read_text()
    msgids = [_f for _f in map(msgid_getter, pot.splitlines()) if _f]

    expected_msgids = [
        "i18n with index entries",
        "index target section",
        "this is :index:`Newsletter` target paragraph.",
        "various index entries",
        "That's all.",
        "Mailing List",
        "Newsletter",
        "Recipients List",
        "First",
        "Second",
        "Third",
        "Entry",
        "See",
        "Module",
        "Keyword",
        "Operator",
        "Object",
        "Exception",
        "Statement",
        "Builtin",
    ]
    for expect in expected_msgids:
        assert expect in msgids
        msgids.remove(expect)

    # unexpected msgid existent
    assert msgids == []


@pytest.mark.sphinx(
    'gettext', testroot='intl', srcdir='gettext',
    confoverrides={'gettext_compact': False,
                   'gettext_additional_targets': []})
def test_gettext_disable_index_entries(app):
    # regression test for #976
    app.builder.build(['index_entries'])

    _msgid_getter = re.compile(r'msgid "(.*)"').search

    def msgid_getter(msgid):
        m = _msgid_getter(msgid)
        if m:
            return m.groups()[0]
        return None

    pot = (app.outdir / 'index_entries.pot').read_text()
    msgids = [_f for _f in map(msgid_getter, pot.splitlines()) if _f]

    expected_msgids = [
        "i18n with index entries",
        "index target section",
        "this is :index:`Newsletter` target paragraph.",
        "various index entries",
        "That's all.",
    ]
    for expect in expected_msgids:
        assert expect in msgids
        msgids.remove(expect)

    # unexpected msgid existent
    assert msgids == []


@pytest.mark.sphinx('gettext', testroot='intl', srcdir='gettext')
def test_gettext_template(app):
    app.builder.build_all()
    assert (app.outdir / 'sphinx.pot').isfile()

    result = (app.outdir / 'sphinx.pot').read_text()
    assert "Welcome" in result
    assert "Sphinx %(version)s" in result


@pytest.mark.sphinx('gettext', testroot='gettext-template')
def test_gettext_template_msgid_order_in_sphinxpot(app):
    app.builder.build_all()
    assert (app.outdir / 'sphinx.pot').isfile()

    result = (app.outdir / 'sphinx.pot').read_text()
    assert re.search(
        ('msgid "Template 1".*'
         'msgid "This is Template 1\\.".*'
         'msgid "Template 2".*'
         'msgid "This is Template 2\\.".*'),
        result,
        flags=re.S)


@pytest.mark.sphinx(
    'gettext', srcdir='root-gettext',
    confoverrides={'gettext_compact': 'documentation'})
def test_build_single_pot(app):
    app.builder.build_all()

    assert (app.outdir / 'documentation.pot').isfile()

    result = (app.outdir / 'documentation.pot').read_text()
    assert re.search(
        ('msgid "Todo".*'
         'msgid "Like footnotes.".*'
         'msgid "The minute.".*'
         'msgid "Generated section".*'),
        result,
        flags=re.S)

import os
def test_custom_locale_translation_repro():
    import tempfile
    import shutil
    from sphinx.application import Sphinx
    from sphinx.builders.gettext import MessageCatalogBuilder

    # Create a temporary directory to simulate the project structure
    with tempfile.TemporaryDirectory() as tmpdir:
        srcdir = os.path.join(tmpdir, 'src')
        confdir = srcdir
        outdir = os.path.join(tmpdir, 'out')
        doctreedir = os.path.join(tmpdir, 'doctrees')
        locale_dir = os.path.join(srcdir, 'locale')
        da_locale_dir = os.path.join(locale_dir, 'da', 'LC_MESSAGES')

        # Create necessary directories and files
        os.makedirs(da_locale_dir)
        os.makedirs(srcdir)

        # Create a simple conf.py
        with open(os.path.join(confdir, 'conf.py'), 'w') as f:
            f.write("""
language = 'da'
gettext_auto_build = True
""")

        # Create a simple index.rst
        with open(os.path.join(srcdir, 'index.rst'), 'w') as f:
            f.write("""
.. figure:: image.png
   :caption: Figur 1

.. code-block:: python
   :caption: Viser 1

   print('Hello World')
""")

        # Create a custom sphinx.po file
        with open(os.path.join(da_locale_dir, 'sphinx.po'), 'w') as f:
            f.write('''\
msgid ""\
msgstr ""\
"Project-Id-Version: PROJECT VERSION\n"POT-Creation-Date: 2023-10-01 00:00+0000\n"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"Last-Translator: FULL NAME <EMAIL@ADDRESS>\n"Language-Team: LANGUAGE <LL@li.org>\n"MIME-Version: 1.0\n"Content-Type: text/plain; charset=UTF-8\n"Content-Transfer-Encoding: 8bit\n"

msgid "Fig. %s"
msgstr "Foobar %s"

msgid "Listing %s"
msgstr "Whatever %s"
''')

        # Build the Sphinx documentation
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername='gettext')
        app.build()

        # Build the MO file from PO file
        subprocess.run(['msgfmt', os.path.join(da_locale_dir, 'sphinx.po'), '-o', os.path.join(da_locale_dir, 'sphinx.mo')], check=True)

        # Rebuild the Sphinx documentation with the MO file
        app = Sphinx(srcdir, confdir, outdir, doctreedir, buildername='html')
        app.build()

        # Check the output files for the translated captions
        with open(os.path.join(outdir, 'index.html'), 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'Foobar 1' in content
            assert 'Whatever 1' in content
