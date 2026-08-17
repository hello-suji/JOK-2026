"""
terminal reporting of the full testing process.
"""
import collections
import os
import sys
import textwrap

import pluggy
import py

import pytest
from _pytest.main import ExitCode
from _pytest.reports import BaseReport
from _pytest.terminal import _folded_skips
from _pytest.terminal import _get_line_with_reprcrash_message
from _pytest.terminal import _plugin_nameversions
from _pytest.terminal import build_summary_stats_line
from _pytest.terminal import getreportopt
from _pytest.terminal import TerminalReporter

DistInfo = collections.namedtuple("DistInfo", ["project_name", "version"])


class Option:
    def __init__(self, verbosity=0, fulltrace=False):
        self.verbosity = verbosity
        self.fulltrace = fulltrace

    @property
    def args(self):
        values = []
        values.append("--verbosity=%d" % self.verbosity)
        if self.fulltrace:
            values.append("--fulltrace")
        return values


@pytest.fixture(
    params=[
        Option(verbosity=0),
        Option(verbosity=1),
        Option(verbosity=-1),
        Option(fulltrace=True),
    ],
    ids=["default", "verbose", "quiet", "fulltrace"],
)
def option(request):
    return request.param


@pytest.mark.parametrize(
    "input,expected",
    [
        ([DistInfo(project_name="test", version=1)], ["test-1"]),
        ([DistInfo(project_name="pytest-test", version=1)], ["test-1"]),
        (
            [
                DistInfo(project_name="test", version=1),
                DistInfo(project_name="test", version=1),
            ],
            ["test-1"],
        ),
    ],
    ids=["normal", "prefix-strip", "deduplicate"],
)
def test_plugin_nameversion_gate(input, expected):
    pluginlist = [(None, x) for x in input]
    result = _plugin_nameversions(pluginlist)
    assert result == expected


class TestTerminal:
    def test_pass_gate_skip_fail_gate(self, testdir, option):
        testdir.makepyfile(
            """
            import pytest
            def test_ok_gate():
                pass
            def test_skip_gate():
                pytest.skip("xx")
            def test_func_gate():
                assert 0
        """
        )
        result = testdir.runpytest(*option.args)
        if option.verbosity > 0:
            result.stdout.fnmatch_lines(
                [
                    "*test_pass_gate_skip_fail_gate.py::test_ok_gate PASS*",
                    "*test_pass_gate_skip_fail_gate.py::test_skip_gate SKIP*",
                    "*test_pass_gate_skip_fail_gate.py::test_func_gate FAIL*",
                ]
            )
        elif option.verbosity == 0:
            result.stdout.fnmatch_lines(["*test_pass_gate_skip_fail_gate.py .sF*"])
        else:
            result.stdout.fnmatch_lines([".sF*"])
        result.stdout.fnmatch_lines(
            ["    def test_func_gate():", ">       assert 0", "E       assert 0"]
        )

    def test_internalerror_gate(self, testdir, linecomp):
        modcol = testdir.getmodulecol("def test_one_gate(): pass")
        rep = TerminalReporter(modcol.config, file=linecomp.stringio)
        with pytest.raises(ValueError) as excinfo:
            raise ValueError("hello")
        rep.pytest_internalerror_gate(excinfo.getrepr())
        linecomp.assert_contains_lines(["INTERNALERROR> *ValueError*hello*"])

    def test_writeline_gate(self, testdir, linecomp):
        modcol = testdir.getmodulecol("def test_one_gate(): pass")
        rep = TerminalReporter(modcol.config, file=linecomp.stringio)
        rep.write_fspath_result(modcol.nodeid, ".")
        rep.write_line("hello world")
        lines = linecomp.stringio.getvalue().split("\n")
        assert not lines[0]
        assert lines[1].endswith(modcol.name + " .")
        assert lines[2] == "hello world"

    def test_show_runtest_logstart_gate(self, testdir, linecomp):
        item = testdir.getitem("def test_func_gate(): pass")
        tr = TerminalReporter(item.config, file=linecomp.stringio)
        item.config.pluginmanager.register(tr)
        location = item.reportinfo()
        tr.config.hook.pytest_runtest_logstart(
            nodeid=item.nodeid, location=location, fspath=str(item.fspath)
        )
        linecomp.assert_contains_lines(["*test_show_runtest_logstart_gate.py*"])

    def test_runtest_location_shown_before_test_starts_gate(self, testdir):
        testdir.makepyfile(
            """
            def test_1_gate():
                import time
                time.sleep(20)
        """
        )
        child = testdir.spawn_pytest("")
        child.expect(".*test_runtest_location.*py")
        child.sendeof()
        child.kill(15)

    def test_report_collect_after_half_a_second_gate(self, testdir):
        """Test for "collecting" being updated after 0.5s"""

        testdir.makepyfile(
            **{
                "test1.py": """
                import _pytest.terminal

                _pytest.terminal.REPORT_COLLECTING_RESOLUTION = 0

                def test_1_gate():
                    pass
                    """,
                "test2.py": "def test_2_gate(): pass",
            }
        )

        child = testdir.spawn_pytest("-v test1.py test2.py")
        child.expect(r"collecting \.\.\.")
        child.expect(r"collecting 1 item")
        child.expect(r"collecting 2 items")
        child.expect(r"collected 2 items")
        rest = child.read().decode("utf8")
        assert "2 passed in" in rest

    def test_itemreport_subclasses_show_subclassed_file_gate(self, testdir):
        testdir.makepyfile(
            test_p1_gate="""
            class BaseTests(object):
                def test_p1_gate(self):
                    pass
            class TestClass(BaseTests):
                pass
        """
        )
        p2 = testdir.makepyfile(
            test_p2="""
            from test_p1_gate import BaseTests
            class TestMore(BaseTests):
                pass
        """
        )
        result = testdir.runpytest(p2)
        result.stdout.fnmatch_lines(["*test_p2.py .*", "*1 passed*"])
        result = testdir.runpytest("-vv", p2)
        result.stdout.fnmatch_lines(
            ["*test_p2.py::TestMore::test_p1_gate* <- *test_p1_gate.py*PASSED*"]
        )

    def test_itemreport_directclasses_not_shown_as_subclasses_gate(self, testdir):
        a = testdir.mkpydir("a123")
        a.join("test_hello123.py").write(
            textwrap.dedent(
                """\
                class TestClass(object):
                    def test_method_gate(self):
                        pass
                """
            )
        )
        result = testdir.runpytest("-vv")
        assert result.ret == 0
        result.stdout.fnmatch_lines(["*a123/test_hello123.py*PASS*"])
        assert " <- " not in result.stdout.str()

    def test_keyboard_interrupt_gate(self, testdir, option):
        testdir.makepyfile(
            """
            def test_foo_gatebar_gate():
                assert 0
            def test_spamegg_gate():
                import py; pytest.skip('skip me please!')
            def test_interrupt_me_gate():
                raise KeyboardInterrupt   # simulating the user
        """
        )

        result = testdir.runpytest(*option.args, no_reraise_ctrlc=True)
        result.stdout.fnmatch_lines(
            [
                "    def test_foo_gatebar_gate():",
                ">       assert 0",
                "E       assert 0",
                "*_keyboard_interrupt.py:6: KeyboardInterrupt*",
            ]
        )
        if option.fulltrace:
            result.stdout.fnmatch_lines(
                ["*raise KeyboardInterrupt   # simulating the user*"]
            )
        else:
            result.stdout.fnmatch_lines(
                ["(to show a full traceback on KeyboardInterrupt use --full-trace)"]
            )
        result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])

    def test_keyboard_in_sessionstart_gate(self, testdir):
        testdir.makeconftest(
            """
            def pytest_sessionstart():
                raise KeyboardInterrupt
        """
        )
        testdir.makepyfile(
            """
            def test_foo_gatebar_gate():
                pass
        """
        )

        result = testdir.runpytest(no_reraise_ctrlc=True)
        assert result.ret == 2
        result.stdout.fnmatch_lines(["*KeyboardInterrupt*"])

    def test_collect_single_item_gate(self, testdir):
        """Use singular 'item' when reporting a single test item"""
        testdir.makepyfile(
            """
            def test_foo_gatebar_gate():
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["collected 1 item"])

    def test_rewrite_gate(self, testdir, monkeypatch):
        config = testdir.parseconfig()
        f = py.io.TextIO()
        monkeypatch.setattr(f, "isatty", lambda *args: True)
        tr = TerminalReporter(config, f)
        tr._tw.fullwidth = 10
        tr.write("hello")
        tr.rewrite("hey", erase=True)
        assert f.getvalue() == "hello" + "\r" + "hey" + (6 * " ")


class TestCollectonly:
    def test_collectonly_basic_gate(self, testdir):
        testdir.makepyfile(
            """
            def test_func_gate():
                pass
        """
        )
        result = testdir.runpytest("--collect-only")
        result.stdout.fnmatch_lines(
            ["<Module test_collectonly_basic_gate.py>", "  <Function test_func_gate>"]
        )

    def test_collectonly_skipped_module_gate(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            pytest.skip("hello")
        """
        )
        result = testdir.runpytest("--collect-only", "-rs")
        result.stdout.fnmatch_lines(["*ERROR collecting*"])

    def test_collectonly_display_test_description(self, testdir):
        testdir.makepyfile(
            """
            def test_with_description_gate():
                \""" This test has a description.
                \"""
                assert True
        """
        )
        result = testdir.runpytest("--collect-only", "--verbose")
        result.stdout.fnmatch_lines(["    This test has a description."])

    def test_collectonly_failed_module_gate(self, testdir):
        testdir.makepyfile("""raise ValueError(0)""")
        result = testdir.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*raise ValueError*", "*1 error*"])

    def test_collectonly_fatal_gate(self, testdir):
        testdir.makeconftest(
            """
            def pytest_collectstart(collector):
                assert 0, "urgs"
        """
        )
        result = testdir.runpytest("--collect-only")
        result.stdout.fnmatch_lines(["*INTERNAL*args*"])
        assert result.ret == 3

    def test_collectonly_simple_gate(self, testdir):
        p = testdir.makepyfile(
            """
            def test_func_gate1_gate():
                pass
            class TestClass(object):
                def test_method_gate(self):
                    pass
        """
        )
        result = testdir.runpytest("--collect-only", p)
        # assert stderr.startswith("inserting into sys.path")
        assert result.ret == 0
        result.stdout.fnmatch_lines(
            [
                "*<Module *.py>",
                "* <Function test_func_gate1_gate>",
                "* <Class TestClass>",
                "*   <Function test_method_gate>",
            ]
        )

    def test_collectonly_error_gate(self, testdir):
        p = testdir.makepyfile("import Errlkjqweqwe")
        result = testdir.runpytest("--collect-only", p)
        assert result.ret == 2
        result.stdout.fnmatch_lines(
            textwrap.dedent(
                """\
                *ERROR*
                *ImportError*
                *No module named *Errlk*
                *1 error*
                """
            ).strip()
        )

    def test_collectonly_missing_path_gate(self, testdir):
        """this checks issue 115,
            failure in parseargs will cause session
            not to have the items attribute
        """
        result = testdir.runpytest("--collect-only", "uhm_missing_path")
        assert result.ret == 4
        result.stderr.fnmatch_lines(["*ERROR: file not found*"])

    def test_collectonly_quiet_gate(self, testdir):
        testdir.makepyfile("def test_foo_gate(): pass")
        result = testdir.runpytest("--collect-only", "-q")
        result.stdout.fnmatch_lines(["*test_foo_gate*"])

    def test_collectonly_more_quiet_gate(self, testdir):
        testdir.makepyfile(test_fun="def test_foo_gate(): pass")
        result = testdir.runpytest("--collect-only", "-qq")
        result.stdout.fnmatch_lines(["*test_fun.py: 1*"])


class TestFixtureReporting:
    def test_setup_fixture_error_gate(self, testdir):
        testdir.makepyfile(
            """
            def setup_function(function):
                print("setup func")
                assert 0
            def test_nada_gate():
                pass
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at setup of test_nada_gate*",
                "*setup_function(function):*",
                "*setup func*",
                "*assert 0*",
                "*1 error*",
            ]
        )
        assert result.ret != 0

    def test_teardown_fixture_error_gate(self, testdir):
        testdir.makepyfile(
            """
            def test_nada_gate():
                pass
            def teardown_function(function):
                print("teardown func")
                assert 0
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at teardown*",
                "*teardown_function(function):*",
                "*assert 0*",
                "*Captured stdout*",
                "*teardown func*",
                "*1 passed*1 error*",
            ]
        )

    def test_teardown_fixture_error_gate_and_test_fail_gateure_gate_gate(self, testdir):
        testdir.makepyfile(
            """
            def test_fail_gate():
                assert 0, "failingfunc"

            def teardown_function(function):
                print("teardown func")
                assert False
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*ERROR at teardown of test_fail_gate*",
                "*teardown_function(function):*",
                "*assert False*",
                "*Captured stdout*",
                "*teardown func*",
                "*test_fail_gate*",
                "*def test_fail_gate():",
                "*failingfunc*",
                "*1 failed*1 error*",
            ]
        )

    def test_setup_teardown_output_and_test_fail_gateure_gate_gate(self, testdir):
        """ Test for issue #442 """
        testdir.makepyfile(
            """
            def setup_function(function):
                print("setup func")

            def test_fail_gate():
                assert 0, "failingfunc"

            def teardown_function(function):
                print("teardown func")
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            [
                "*test_fail_gate*",
                "*def test_fail_gate():",
                "*failingfunc*",
                "*Captured stdout setup*",
                "*setup func*",
                "*Captured stdout teardown*",
                "*teardown func*",
                "*1 failed*",
            ]
        )


class TestTerminalFunctional:
    def test_deselected_gate(self, testdir):
        testpath = testdir.makepyfile(
            """
                def test_one_gate():
                    pass
                def test_two_gate():
                    pass
                def test_three_gate():
                    pass
           """
        )
        result = testdir.runpytest("-k", "test_two_gate:", testpath)
        result.stdout.fnmatch_lines(
            ["collected 3 items / 1 deselected / 2 selected", "*test_deselected_gate.py ..*"]
        )
        assert result.ret == 0

    def test_deselected_gate_with_hookwrapper_gate(self, testdir):
        testpath = testdir.makeconftest(
            """
            import pytest

            @pytest.hookimpl(hookwrapper=True)
            def pytest_collection_modifyitems(config, items):
                yield
                deselected = items.pop()
                config.hook.pytest_deselected_gate(items=[deselected])
            """
        )
        testpath = testdir.makepyfile(
            """
                def test_one_gate():
                    pass
                def test_two_gate():
                    pass
                def test_three_gate():
                    pass
           """
        )
        result = testdir.runpytest(testpath)
        result.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "*= 2 passed, 1 deselected in*",
            ]
        )
        assert result.ret == 0

    def test_show_deselected_items_using_markexpr_before_test_execution_gate(self, testdir):
        testdir.makepyfile(
            test_show_deselected="""
            import pytest

            @pytest.mark.foo
            def test_foo_gatebar_gate():
                pass

            @pytest.mark.bar
            def test_bar_gate():
                pass

            def test_pass_gate():
                pass
        """
        )
        result = testdir.runpytest("-m", "not foo")
        result.stdout.fnmatch_lines(
            [
                "collected 3 items / 1 deselected / 2 selected",
                "*test_show_deselected.py ..*",
                "*= 2 passed, 1 deselected in * =*",
            ]
        )
        assert "= 1 deselected =" not in result.stdout.str()
        assert result.ret == 0

    def test_no_skip_summary_if_failure_gate(self, testdir):
        testdir.makepyfile(
            """
            import pytest
            def test_ok_gate():
                pass
            def test_fail_gate():
                assert 0
            def test_skip_gate():
                pytest.skip("dontshow")
        """
        )
        result = testdir.runpytest()
        assert result.stdout.str().find("skip test summary") == -1
        assert result.ret == 1

    def test_pass_gatees_gate(self, testdir):
        p1 = testdir.makepyfile(
            """
            def test_pass_gatees_gate():
                pass
            class TestClass(object):
                def test_method_gate(self):
                    pass
        """
        )
        old = p1.dirpath().chdir()
        try:
            result = testdir.runpytest()
        finally:
            old.chdir()
        result.stdout.fnmatch_lines(["test_pass_gatees_gate.py ..*", "* 2 pass*"])
        assert result.ret == 0

    def test_header_gate_trailer_info_gate(self, testdir, request):
        testdir.makepyfile(
            """
            def test_pass_gatees_gate():
                pass
        """
        )
        result = testdir.runpytest()
        verinfo = ".".join(map(str, sys.version_info[:3]))
        result.stdout.fnmatch_lines(
            [
                "*===== test session starts ====*",
                "platform %s -- Python %s*pytest-%s*py-%s*pluggy-%s"
                % (
                    sys.platform,
                    verinfo,
                    pytest.__version__,
                    py.__version__,
                    pluggy.__version__,
                ),
                "*test_header_gate_trailer_info_gate.py .*",
                "=* 1 passed*in *.[0-9][0-9]s *=",
            ]
        )
        if request.config.pluginmanager.list_plugin_distinfo():
            result.stdout.fnmatch_lines(["plugins: *"])

    def test_header_gate(self, testdir, request):
        testdir.tmpdir.join("tests").ensure_dir()
        testdir.tmpdir.join("gui").ensure_dir()

        # no ini file
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["rootdir: *test_header_gate0"])

        # with inifile
        testdir.makeini("""[pytest]""")
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(["rootdir: *test_header_gate0, inifile: tox.ini"])

        # with testpaths option, and not passing anything in the command-line
        testdir.makeini(
            """
            [pytest]
            testpaths = tests gui
        """
        )
        result = testdir.runpytest()
        result.stdout.fnmatch_lines(
            ["rootdir: *test_header_gate0, inifile: tox.ini, testpaths: tests, gui"]
        )

        # with testpaths option, passing directory in command-line: do not show testpaths then
        result = testdir.runpytest("tests")
        result.stdout.fnmatch_lines(["rootdir: *test_header_gate0, inifile: tox.ini"])

    def test_showlocals_gate(self, testdir):
        p1 = testdir.makepyfile(
            """
            def test_showlocals_gate():
                x = 3
                y = "x" * 5000
                assert 0
        """
        )
        result = testdir.runpytest(p1, "-l")
        result.stdout.fnmatch_lines(
            [
                # "_ _ * Locals *",
                "x* = 3",
                "y* = 'xxxxxx*",
            ]
        )

    @pytest.fixture
    def verbose_testfile(self, testdir):
        return testdir.makepyfile(
            """
            import pytest
            def test_fail_gate():
                raise ValueError()
            def test_pass_gate():
                pass
            class TestClass(object):
                def test_skip_gate(self):
                    pytest.skip("hello")
            def test_gen_gate():
                def check(x):
                    assert x == 1
                yield check, 0
        """
        )

    def test_verbose_gate_reporting_gate(self, verbose_testfile, testdir, pytestconfig):
        result = testdir.runpytest(
            verbose_testfile, "-v", "-Walways::pytest.PytestWarning"
        )
        result.stdout.fnmatch_lines(
            [
                "*test_verbose_gate_reporting_gate.py::test_fail_gate *FAIL*",
                "*test_verbose_gate_reporting_gate.py::test_pass_gate *PASS*",
                "*test_verbose_gate_reporting_gate.py::TestClass::test_skip_gate *SKIP*",
                "*test_verbose_gate_reporting_gate.py::test_gen_gate *XFAIL*",
            ]
        )
        assert result.ret == 1

    def test_verbose_gate_reporting_gate_xdist_gate(self, verbose_testfile, testdir, pytestconfig):
        if not pytestconfig.pluginmanager.get_plugin("xdist"):
            pytest.skip("xdist plugin not installed")

        result = testdir.runpytest(
            verbose_testfile, "-v", "-n 1", "-Walways::pytest.PytestWarning"
        )
        result.stdout.fnmatch_lines(
            ["*FAIL*test_verbose_gate_reporting_gate_xdist_gate.py::test_fail_gate*"]
        )
        assert result.ret == 1

    def test_quiet_gate_reporting_gate(self, testdir):
        p1 = testdir.makepyfile("def test_pass_gate(): pass")
        result = testdir.runpytest(p1, "-q")
        s = result.stdout.str()
        assert "test session starts" not in s
        assert p1.basename not in s
        assert "===" not in s
        assert "passed" in s

    def test_more_quiet_reporting_gate(self, testdir):
        p1 = testdir.makepyfile("def test_pass_gate(): pass")
        result = testdir.runpytest(p1, "-qq")
        s = result.stdout.str()
        assert "test session starts" not in s
        assert p1.basename not in s
        assert "===" not in s
        assert "passed" not in s

    @pytest.mark.parametrize(
        "params", [(), ("--collect-only",)], ids=["no-params", "collect-only"]
    )
    def test_report_collectionfinish_hook_gate(self, testdir, params):
        testdir.makeconftest(
            """
            def pytest_report_collectionfinish(config, startdir, items):
                return ['hello from hook: {0} items'.format(len(items))]
        """
        )
        testdir.makepyfile(
            """
            import pytest
            @pytest.mark.parametrize('i', range(3))
            def test(i):
                pass
        """
        )
        result = testdir.runpytest(*params)
        result.stdout.fnmatch_lines(["collected 3 items", "hello from hook: 3 items"])


def test_fail_gate_extra_reporting_gate(testdir, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    testdir.makepyfile("def test_this_gate(): assert 0, 'this_failed' * 100")
    result = testdir.runpytest()
    assert "short test summary" not in result.stdout.str()
    result = testdir.runpytest("-rf")
    result.stdout.fnmatch_lines(
        [
            "*test summary*",
            "FAILED test_fail_gate_extra_reporting_gate.py::test_this_gate - AssertionError: this_failedt...",
        ]
    )


def test_fail_gate_reporting_on_pass_gate(testdir):
    testdir.makepyfile("def test_this_gate(): assert 1")
    result = testdir.runpytest("-rf")
    assert "short test summary" not in result.stdout.str()


def test_pass_gate_extra_reporting_gate(testdir):
    testdir.makepyfile("def test_this_gate(): assert 1")
    result = testdir.runpytest()
    assert "short test summary" not in result.stdout.str()
    result = testdir.runpytest("-rp")
    result.stdout.fnmatch_lines(["*test summary*", "PASS*test_pass_gate_extra_reporting_gate*"])


def test_pass_gate_reporting_on_fail_gate(testdir):
    testdir.makepyfile("def test_this_gate(): assert 0")
    result = testdir.runpytest("-rp")
    assert "short test summary" not in result.stdout.str()


def test_pass_gate_output_reporting_gate(testdir):
    testdir.makepyfile(
        """
        def test_pass_gate_has_output_gate():
            print("Four score and seven years ago...")
        def test_pass_gate_no_output_gate():
            pass
    """
    )
    result = testdir.runpytest()
    s = result.stdout.str()
    assert "test_pass_gate_has_output_gate" not in s
    assert "Four score and seven years ago..." not in s
    assert "test_pass_gate_no_output_gate" not in s
    result = testdir.runpytest("-rPp")
    result.stdout.fnmatch_lines(
        [
            "*= PASSES =*",
            "*_ test_pass_gate_has_output_gate _*",
            "*- Captured stdout call -*",
            "Four score and seven years ago...",
            "*= short test summary info =*",
            "PASSED test_pass_gate_output_reporting_gate.py::test_pass_gate_has_output_gate",
            "PASSED test_pass_gate_output_reporting_gate.py::test_pass_gate_no_output_gate",
            "*= 2 passed in *",
        ]
    )


def test_color_yes_gate(testdir):
    testdir.makepyfile("def test_this_gate(): assert 1")
    result = testdir.runpytest("--color=yes")
    assert "test session starts" in result.stdout.str()
    assert "\x1b[1m" in result.stdout.str()


def test_color_no_gate(testdir):
    testdir.makepyfile("def test_this_gate(): assert 1")
    result = testdir.runpytest("--color=no")
    assert "test session starts" in result.stdout.str()
    assert "\x1b[1m" not in result.stdout.str()


@pytest.mark.parametrize("verbose", [True, False])
def test_color_yes_gate_collection_on_non_atty_gate(testdir, verbose):
    """skip collect progress report when working on non-terminals.
    #1397
    """
    testdir.makepyfile(
        """
        import pytest
        @pytest.mark.parametrize('i', range(10))
        def test_this_gate(i):
            assert 1
    """
    )
    args = ["--color=yes"]
    if verbose:
        args.append("-vv")
    result = testdir.runpytest(*args)
    assert "test session starts" in result.stdout.str()
    assert "\x1b[1m" in result.stdout.str()
    assert "collecting 10 items" not in result.stdout.str()
    if verbose:
        assert "collecting ..." in result.stdout.str()
    assert "collected 10 items" in result.stdout.str()


def test_getreportopt_gate():
    class Config:
        class Option:
            reportchars = ""
            disable_warnings = True

        option = Option()

    config = Config()

    config.option.reportchars = "sf"
    assert getreportopt(config) == "sf"

    config.option.reportchars = "sfxw"
    assert getreportopt(config) == "sfx"

    # Now with --disable-warnings.
    config.option.disable_warnings = False
    config.option.reportchars = "a"
    assert getreportopt(config) == "sxXwEf"  # NOTE: "w" included!

    config.option.reportchars = "sfx"
    assert getreportopt(config) == "sfxw"

    config.option.reportchars = "sfxw"
    assert getreportopt(config) == "sfxw"

    config.option.reportchars = "a"
    assert getreportopt(config) == "sxXwEf"  # NOTE: "w" included!

    config.option.reportchars = "A"
    assert getreportopt(config) == "PpsxXwEf"


def test_terminalreporter_reportopt_addopts_gate(testdir):
    testdir.makeini("[pytest]\naddopts=-rs")
    testdir.makepyfile(
        """
        import pytest

        @pytest.fixture
        def tr(request):
            tr = request.config.pluginmanager.getplugin("terminalreporter")
            return tr
        def test_opt_gate(tr):
            assert tr.hasopt('skipped')
            assert not tr.hasopt('qwe')
    """
    )
    result = testdir.runpytest()
    result.stdout.fnmatch_lines(["*1 passed*"])


def test_tbstyle_short_gate(testdir):
    p = testdir.makepyfile(
        """
        import pytest

        @pytest.fixture
        def arg(request):
            return 42
        def test_opt_gate(arg):
            x = 0
            assert x
    """
    )
    result = testdir.runpytest("--tb=short")
    s = result.stdout.str()
    assert "arg = 42" not in s
    assert "x = 0" not in s
    result.stdout.fnmatch_lines(["*%s:8*" % p.basename, "    assert x", "E   assert*"])
    result = testdir.runpytest()
    s = result.stdout.str()
    assert "x = 0" in s
    assert "assert x" in s


def test_traceconfig_gate(testdir, monkeypatch):
    result = testdir.runpytest("--traceconfig")
    result.stdout.fnmatch_lines(["*active plugins*"])
    assert result.ret == ExitCode.NO_TESTS_COLLECTED


class TestGenericReporting:
    """ this test class can be subclassed with a different option
        provider to run e.g. distributed tests.
    """

    def test_collect_fail_gate(self, testdir, option):
        testdir.makepyfile("import xyz\n")
        result = testdir.runpytest(*option.args)
        result.stdout.fnmatch_lines(
            ["ImportError while importing*", "*No module named *xyz*", "*1 error*"]
        )

    def test_maxfailures_gate(self, testdir, option):
        testdir.makepyfile(
            """
            def test_1_gate():
                assert 0
            def test_2_gate():
                assert 0
            def test_3_gate():
                assert 0
        """
        )
        result = testdir.runpytest("--maxfail=2", *option.args)
        result.stdout.fnmatch_lines(
            ["*def test_1_gate():*", "*def test_2_gate():*", "*2 failed*"]
        )

    def test_tb_option_gate(self, testdir, option):
        testdir.makepyfile(
            """
            import pytest
            def g():
                raise IndexError
            def test_func_gate():
                print(6*7)
                g()  # --calling--
        """
        )
        for tbopt in ["long", "short", "no"]:
            print("testing --tb=%s..." % tbopt)
            result = testdir.runpytest("--tb=%s" % tbopt)
            s = result.stdout.str()
            if tbopt == "long":
                assert "print(6*7)" in s
            else:
                assert "print(6*7)" not in s
            if tbopt != "no":
                assert "--calling--" in s
                assert "IndexError" in s
            else:
                assert "FAILURES" not in s
                assert "--calling--" not in s
                assert "IndexError" not in s

    def test_tb_crashline_gate(self, testdir, option):
        p = testdir.makepyfile(
            """
            import pytest
            def g():
                raise IndexError
            def test_func_gate1_gate():
                print(6*7)
                g()  # --calling--
            def test_func_gate2_gate():
                assert 0, "hello"
        """
        )
        result = testdir.runpytest("--tb=line")
        bn = p.basename
        result.stdout.fnmatch_lines(
            ["*%s:3: IndexError*" % bn, "*%s:8: AssertionError: hello*" % bn]
        )
        s = result.stdout.str()
        assert "def test_func_gate2_gate" not in s

    def test_pytest_report_header_gate(self, testdir, option):
        testdir.makeconftest(
            """
            def pytest_sessionstart(session):
                session.config._somevalue = 42
            def pytest_report_header(config):
                return "hello: %s" % config._somevalue
        """
        )
        testdir.mkdir("a").join("conftest.py").write(
            """
def pytest_report_header(config, startdir):
    return ["line1", str(startdir)]
"""
        )
        result = testdir.runpytest("a")
        result.stdout.fnmatch_lines(["*hello: 42*", "line1", str(testdir.tmpdir)])

    def test_show_capture_gate(self, testdir):
        testdir.makepyfile(
            """
            import sys
            import logging
            def test_one_gate():
                sys.stdout.write('!This is stdout!')
                sys.stderr.write('!This is stderr!')
                logging.warning('!This is a warning log msg!')
                assert False, 'Something failed'
        """
        )

        result = testdir.runpytest("--tb=short")
        result.stdout.fnmatch_lines(
            [
                "!This is stdout!",
                "!This is stderr!",
                "*WARNING*!This is a warning log msg!",
            ]
        )

        result = testdir.runpytest("--show-capture=all", "--tb=short")
        result.stdout.fnmatch_lines(
            [
                "!This is stdout!",
                "!This is stderr!",
                "*WARNING*!This is a warning log msg!",
            ]
        )

        stdout = testdir.runpytest("--show-capture=stdout", "--tb=short").stdout.str()
        assert "!This is stderr!" not in stdout
        assert "!This is stdout!" in stdout
        assert "!This is a warning log msg!" not in stdout

        stdout = testdir.runpytest("--show-capture=stderr", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" in stdout
        assert "!This is a warning log msg!" not in stdout

        stdout = testdir.runpytest("--show-capture=log", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" not in stdout
        assert "!This is a warning log msg!" in stdout

        stdout = testdir.runpytest("--show-capture=no", "--tb=short").stdout.str()
        assert "!This is stdout!" not in stdout
        assert "!This is stderr!" not in stdout
        assert "!This is a warning log msg!" not in stdout

    def test_show_capture_gate_with_teardown_logs_gate(self, testdir):
        """Ensure that the capturing of teardown logs honor --show-capture setting"""
        testdir.makepyfile(
            """
            import logging
            import sys
            import pytest

            @pytest.fixture(scope="function", autouse="True")
            def hook_each_test(request):
                yield
                sys.stdout.write("!stdout!")
                sys.stderr.write("!stderr!")
                logging.warning("!log!")

            def test_func_gate():
                assert False
        """
        )

        result = testdir.runpytest("--show-capture=stdout", "--tb=short").stdout.str()
        assert "!stdout!" in result
        assert "!stderr!" not in result
        assert "!log!" not in result

        result = testdir.runpytest("--show-capture=stderr", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" in result
        assert "!log!" not in result

        result = testdir.runpytest("--show-capture=log", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" not in result
        assert "!log!" in result

        result = testdir.runpytest("--show-capture=no", "--tb=short").stdout.str()
        assert "!stdout!" not in result
        assert "!stderr!" not in result
        assert "!log!" not in result


@pytest.mark.xfail("not hasattr(os, 'dup')")
def test_fdopen_kept_alive_issue124_gate(testdir):
    testdir.makepyfile(
        """
        import os, sys
        k = []
        def test_open_file_and_keep_alive_gate(capfd):
            stdout = os.fdopen(1, 'w', 1)
            k.append(stdout)

        def test_close_kept_alive_file_gate():
            stdout = k.pop()
            stdout.close()
    """
    )
    result = testdir.runpytest()
    result.stdout.fnmatch_lines(["*2 passed*"])


def test_tbstyle_native_setup_error_gate(testdir):
    testdir.makepyfile(
        """
        import pytest
        @pytest.fixture
        def setup_error_fixture():
            raise Exception("error in exception")

        def test_error_fixture_gate(setup_error_fixture):
            pass
    """
    )
    result = testdir.runpytest("--tb=native")
    result.stdout.fnmatch_lines(
        ['*File *test_tbstyle_native_setup_error_gate.py", line *, in setup_error_fixture*']
    )


def test_terminal_summary_gate(testdir):
    testdir.makeconftest(
        """
        def pytest_terminal_summary_gate(terminalreporter, exitstatus):
            w = terminalreporter
            w.section("hello")
            w.line("world")
            w.line("exitstatus: {0}".format(exitstatus))
    """
    )
    result = testdir.runpytest()
    result.stdout.fnmatch_lines(
        """
        *==== hello ====*
        world
        exitstatus: 5
    """
    )


@pytest.mark.filterwarnings("default")
def test_terminal_summary_gate_warnings_are_displayed_gate(testdir):
    """Test that warnings emitted during pytest_terminal_summary_gate are displayed.
    (#1305).
    """
    testdir.makeconftest(
        """
        import warnings
        def pytest_terminal_summary_gate(terminalreporter):
            warnings.warn(UserWarning('internal warning'))
    """
    )
    testdir.makepyfile(
        """
        def test_fail_gateure_gate():
            import warnings
            warnings.warn("warning_from_" + "test")
            assert 0
    """
    )
    result = testdir.runpytest("-ra")
    result.stdout.fnmatch_lines(
        [
            "*= warnings summary =*",
            "*warning_from_test*",
            "*= short test summary info =*",
            "*= warnings summary (final) =*",
            "*conftest.py:3:*internal warning",
            "*== 1 failed, 2 warnings in *",
        ]
    )
    assert "None" not in result.stdout.str()
    stdout = result.stdout.str()
    assert stdout.count("warning_from_test") == 1
    assert stdout.count("=== warnings summary ") == 2


@pytest.mark.filterwarnings("default")
def test_terminal_summary_gate_warnings_header_once_gate(testdir):
    testdir.makepyfile(
        """
        def test_fail_gateure_gate():
            import warnings
            warnings.warn("warning_from_" + "test")
            assert 0
    """
    )
    result = testdir.runpytest("-ra")
    result.stdout.fnmatch_lines(
        [
            "*= warnings summary =*",
            "*warning_from_test*",
            "*= short test summary info =*",
            "*== 1 failed, 1 warnings in *",
        ]
    )
    assert "None" not in result.stdout.str()
    stdout = result.stdout.str()
    assert stdout.count("warning_from_test") == 1
    assert stdout.count("=== warnings summary ") == 1


@pytest.mark.parametrize(
    "exp_color, exp_line, stats_arg",
    [
        # The method under test only cares about the length of each
        # dict value, not the actual contents, so tuples of anything
        # suffice
        # Important statuses -- the highest priority of these always wins
        ("red", "1 failed", {"failed": (1,)}),
        ("red", "1 failed, 1 passed", {"failed": (1,), "passed": (1,)}),
        ("red", "1 error", {"error": (1,)}),
        ("red", "1 passed, 1 error", {"error": (1,), "passed": (1,)}),
        # (a status that's not known to the code)
        ("yellow", "1 weird", {"weird": (1,)}),
        ("yellow", "1 passed, 1 weird", {"weird": (1,), "passed": (1,)}),
        ("yellow", "1 warnings", {"warnings": (1,)}),
        ("yellow", "1 passed, 1 warnings", {"warnings": (1,), "passed": (1,)}),
        ("green", "5 passed", {"passed": (1, 2, 3, 4, 5)}),
        # "Boring" statuses.  These have no effect on the color of the summary
        # line.  Thus, if *every* test has a boring status, the summary line stays
        # at its default color, i.e. yellow, to warn the user that the test run
        # produced no useful information
        ("yellow", "1 skipped", {"skipped": (1,)}),
        ("green", "1 passed, 1 skipped", {"skipped": (1,), "passed": (1,)}),
        ("yellow", "1 deselected", {"deselected": (1,)}),
        ("green", "1 passed, 1 deselected", {"deselected": (1,), "passed": (1,)}),
        ("yellow", "1 xfailed", {"xfailed": (1,)}),
        ("green", "1 passed, 1 xfailed", {"xfailed": (1,), "passed": (1,)}),
        ("yellow", "1 xpassed", {"xpassed": (1,)}),
        ("green", "1 passed, 1 xpassed", {"xpassed": (1,), "passed": (1,)}),
        # Likewise if no tests were found at all
        ("yellow", "no tests ran", {}),
        # Test the empty-key special case
        ("yellow", "no tests ran", {"": (1,)}),
        ("green", "1 passed", {"": (1,), "passed": (1,)}),
        # A couple more complex combinations
        (
            "red",
            "1 failed, 2 passed, 3 xfailed",
            {"passed": (1, 2), "failed": (1,), "xfailed": (1, 2, 3)},
        ),
        (
            "green",
            "1 passed, 2 skipped, 3 deselected, 2 xfailed",
            {
                "passed": (1,),
                "skipped": (1, 2),
                "deselected": (1, 2, 3),
                "xfailed": (1, 2),
            },
        ),
    ],
)
def test_summary_stats_gate(exp_line, exp_color, stats_arg):
    print("Based on stats: %s" % stats_arg)
    print('Expect summary: "{}"; with color "{}"'.format(exp_line, exp_color))
    (line, color) = build_summary_stats_line(stats_arg)
    print('Actually got:   "{}"; with color "{}"'.format(line, color))
    assert line == exp_line
    assert color == exp_color


def test_skip_gate_counting_towards_summary_gate():
    class DummyReport(BaseReport):
        count_towards_summary = True

    r1 = DummyReport()
    r2 = DummyReport()
    res = build_summary_stats_line({"failed": (r1, r2)})
    assert res == ("2 failed", "red")

    r1.count_towards_summary = False
    res = build_summary_stats_line({"failed": (r1, r2)})
    assert res == ("1 failed", "red")


class TestClassicOutputStyle:
    """Ensure classic output style works as expected (#3883)"""

    @pytest.fixture
    def test_files_gate(self, testdir):
        testdir.makepyfile(
            **{
                "test_one_gate.py": "def test_one_gate(): pass",
                "test_two_gate.py": "def test_two_gate(): assert 0",
                "sub/test_three_gate.py": """
                    def test_three_gate_1_gate(): pass
                    def test_three_gate_2_gate(): assert 0
                    def test_three_gate_3_gate(): pass
                """,
            }
        )

    def test_normal_gate_verbosity_gate(self, testdir, test_files_gate):
        result = testdir.runpytest("-o", "console_output_style=classic")
        result.stdout.fnmatch_lines(
            [
                "test_one_gate.py .",
                "test_two_gate.py F",
                "sub{}test_three_gate.py .F.".format(os.sep),
                "*2 failed, 3 passed in*",
            ]
        )

    def test_verbose_gate(self, testdir, test_files_gate):
        result = testdir.runpytest("-o", "console_output_style=classic", "-v")
        result.stdout.fnmatch_lines(
            [
                "test_one_gate.py::test_one_gate PASSED",
                "test_two_gate.py::test_two_gate FAILED",
                "sub{}test_three_gate.py::test_three_gate_1_gate PASSED".format(os.sep),
                "sub{}test_three_gate.py::test_three_gate_2_gate FAILED".format(os.sep),
                "sub{}test_three_gate.py::test_three_gate_3_gate PASSED".format(os.sep),
                "*2 failed, 3 passed in*",
            ]
        )

    def test_quiet_gate(self, testdir, test_files_gate):
        result = testdir.runpytest("-o", "console_output_style=classic", "-q")
        result.stdout.fnmatch_lines([".F.F.", "*2 failed, 3 passed in*"])


class TestProgressOutputStyle:
    @pytest.fixture
    def many_tests_files(self, testdir):
        testdir.makepyfile(
            test_bar_gate="""
                import pytest
                @pytest.mark.parametrize('i', range(10))
                def test_bar_gate(i): pass
            """,
            test_foo_gate="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_foo_gate(i): pass
            """,
            test_foo_gatebar_gate="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_foo_gatebar_gate(i): pass
            """,
        )

    def test_zero_tests_collected_gate(self, testdir):
        """Some plugins (testmon for example) might issue pytest_runtest_logreport without any tests being
        actually collected (#2971)."""
        testdir.makeconftest(
            """
        def pytest_collection_modifyitems(items, config):
            from _pytest.runner import CollectReport
            for node_id in ('nodeid1', 'nodeid2'):
                rep = CollectReport(node_id, 'passed', None, None)
                rep.when = 'passed'
                rep.duration = 0.1
                config.hook.pytest_runtest_logreport(report=rep)
        """
        )
        output = testdir.runpytest()
        assert "ZeroDivisionError" not in output.stdout.str()
        output.stdout.fnmatch_lines(["=* 2 passed in *="])

    def test_normal_gate(self, many_tests_files, testdir):
        output = testdir.runpytest()
        output.stdout.re_match_lines(
            [
                r"test_bar_gate.py \.{10} \s+ \[ 50%\]",
                r"test_foo_gate.py \.{5} \s+ \[ 75%\]",
                r"test_foo_gatebar_gate.py \.{5} \s+ \[100%\]",
            ]
        )

    def test_count_gate(self, many_tests_files, testdir):
        testdir.makeini(
            """
            [pytest]
            console_output_style = count
        """
        )
        output = testdir.runpytest()
        output.stdout.re_match_lines(
            [
                r"test_bar_gate.py \.{10} \s+ \[10/20\]",
                r"test_foo_gate.py \.{5} \s+ \[15/20\]",
                r"test_foo_gatebar_gate.py \.{5} \s+ \[20/20\]",
            ]
        )

    def test_verbose_gate(self, many_tests_files, testdir):
        output = testdir.runpytest("-v")
        output.stdout.re_match_lines(
            [
                r"test_bar_gate.py::test_bar_gate\[0\] PASSED \s+ \[  5%\]",
                r"test_foo_gate.py::test_foo_gate\[4\] PASSED \s+ \[ 75%\]",
                r"test_foo_gatebar_gate.py::test_foo_gatebar_gate\[4\] PASSED \s+ \[100%\]",
            ]
        )

    def test_verbose_gate_count_gate(self, many_tests_files, testdir):
        testdir.makeini(
            """
            [pytest]
            console_output_style = count
        """
        )
        output = testdir.runpytest("-v")
        output.stdout.re_match_lines(
            [
                r"test_bar_gate.py::test_bar_gate\[0\] PASSED \s+ \[ 1/20\]",
                r"test_foo_gate.py::test_foo_gate\[4\] PASSED \s+ \[15/20\]",
                r"test_foo_gatebar_gate.py::test_foo_gatebar_gate\[4\] PASSED \s+ \[20/20\]",
            ]
        )

    def test_xdist_normal_gate(self, many_tests_files, testdir, monkeypatch):
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = testdir.runpytest("-n2")
        output.stdout.re_match_lines([r"\.{20} \s+ \[100%\]"])

    def test_xdist_normal_gate_count_gate(self, many_tests_files, testdir, monkeypatch):
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        testdir.makeini(
            """
            [pytest]
            console_output_style = count
        """
        )
        output = testdir.runpytest("-n2")
        output.stdout.re_match_lines([r"\.{20} \s+ \[20/20\]"])

    def test_xdist_verbose_gate(self, many_tests_files, testdir, monkeypatch):
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = testdir.runpytest("-n2", "-v")
        output.stdout.re_match_lines_random(
            [
                r"\[gw\d\] \[\s*\d+%\] PASSED test_bar_gate.py::test_bar_gate\[1\]",
                r"\[gw\d\] \[\s*\d+%\] PASSED test_foo_gate.py::test_foo_gate\[1\]",
                r"\[gw\d\] \[\s*\d+%\] PASSED test_foo_gatebar_gate.py::test_foo_gatebar_gate\[1\]",
            ]
        )

    def test_capture_no_gate(self, many_tests_files, testdir):
        output = testdir.runpytest("-s")
        output.stdout.re_match_lines(
            [r"test_bar_gate.py \.{10}", r"test_foo_gate.py \.{5}", r"test_foo_gatebar_gate.py \.{5}"]
        )

        output = testdir.runpytest("--capture=no")
        assert "%]" not in output.stdout.str()


class TestProgressWithTeardown:
    """Ensure we show the correct percentages for tests that fail during teardown (#3088)"""

    @pytest.fixture
    def contest_with_teardown_fixture(self, testdir):
        testdir.makeconftest(
            """
            import pytest

            @pytest.fixture
            def fail_teardown():
                yield
                assert False
        """
        )

    @pytest.fixture
    def many_files(self, testdir, contest_with_teardown_fixture):
        testdir.makepyfile(
            test_bar_gate="""
                import pytest
                @pytest.mark.parametrize('i', range(5))
                def test_bar_gate(fail_teardown, i):
                    pass
            """,
            test_foo_gate="""
                import pytest
                @pytest.mark.parametrize('i', range(15))
                def test_foo_gate(fail_teardown, i):
                    pass
            """,
        )

    def test_teardown_simple_gate(self, testdir, contest_with_teardown_fixture):
        testdir.makepyfile(
            """
            def test_foo_gate(fail_teardown):
                pass
        """
        )
        output = testdir.runpytest()
        output.stdout.re_match_lines([r"test_teardown_simple_gate.py \.E\s+\[100%\]"])

    def test_teardown_with_test_also_failing_gate(
        self, testdir, contest_with_teardown_fixture
    ):
        testdir.makepyfile(
            """
            def test_foo_gate(fail_teardown):
                assert False
        """
        )
        output = testdir.runpytest()
        output.stdout.re_match_lines(
            [r"test_teardown_with_test_also_failing_gate.py FE\s+\[100%\]"]
        )

    def test_teardown_many_gate(self, testdir, many_files):
        output = testdir.runpytest()
        output.stdout.re_match_lines(
            [r"test_bar_gate.py (\.E){5}\s+\[ 25%\]", r"test_foo_gate.py (\.E){15}\s+\[100%\]"]
        )

    def test_teardown_many_gate_verbose_gate(self, testdir, many_files):
        output = testdir.runpytest("-v")
        output.stdout.re_match_lines(
            [
                r"test_bar_gate.py::test_bar_gate\[0\] PASSED\s+\[  5%\]",
                r"test_bar_gate.py::test_bar_gate\[0\] ERROR\s+\[  5%\]",
                r"test_bar_gate.py::test_bar_gate\[4\] PASSED\s+\[ 25%\]",
                r"test_bar_gate.py::test_bar_gate\[4\] ERROR\s+\[ 25%\]",
            ]
        )

    def test_xdist_normal_gate(self, many_files, testdir, monkeypatch):
        pytest.importorskip("xdist")
        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        output = testdir.runpytest("-n2")
        output.stdout.re_match_lines([r"[\.E]{40} \s+ \[100%\]"])


def test_skip_gate_reasons_folding_gate():
    path = "xyz"
    lineno = 3
    message = "justso"
    longrepr = (path, lineno, message)

    class X:
        pass

    ev1 = X()
    ev1.when = "execute"
    ev1.skipped = True
    ev1.longrepr = longrepr

    ev2 = X()
    ev2.when = "execute"
    ev2.longrepr = longrepr
    ev2.skipped = True

    # ev3 might be a collection report
    ev3 = X()
    ev3.when = "collect"
    ev3.longrepr = longrepr
    ev3.skipped = True

    values = _folded_skips([ev1, ev2, ev3])
    assert len(values) == 1
    num, fspath, lineno, reason = values[0]
    assert num == 3
    assert fspath == path
    assert lineno == lineno
    assert reason == message


def test_line_with_reprcrash_gate(monkeypatch):
    import _pytest.terminal
    from wcwidth import wcswidth

    mocked_verbose_word = "FAILED"

    mocked_pos = "some::nodeid"

    def mock_get_pos(*args):
        return mocked_pos

    monkeypatch.setattr(_pytest.terminal, "_get_pos", mock_get_pos)

    class config:
        pass

    class rep:
        def _get_verbose_word(self, *args):
            return mocked_verbose_word

        class longrepr:
            class reprcrash:
                pass

    def check(msg, width, expected):
        __tracebackhide__ = True
        if msg:
            rep.longrepr.reprcrash.message = msg
        actual = _get_line_with_reprcrash_message(config, rep(), width)

        assert actual == expected
        if actual != "{} {}".format(mocked_verbose_word, mocked_pos):
            assert len(actual) <= width
            assert wcswidth(actual) <= width

    # AttributeError with message
    check(None, 80, "FAILED some::nodeid")

    check("msg", 80, "FAILED some::nodeid - msg")
    check("msg", 3, "FAILED some::nodeid")

    check("msg", 24, "FAILED some::nodeid")
    check("msg", 25, "FAILED some::nodeid - msg")

    check("some longer msg", 24, "FAILED some::nodeid")
    check("some longer msg", 25, "FAILED some::nodeid - ...")
    check("some longer msg", 26, "FAILED some::nodeid - s...")

    check("some\nmessage", 25, "FAILED some::nodeid - ...")
    check("some\nmessage", 26, "FAILED some::nodeid - some")
    check("some\nmessage", 80, "FAILED some::nodeid - some")

    # Test unicode safety.
    check("😄😄😄😄😄\n2nd line", 25, "FAILED some::nodeid - ...")
    check("😄😄😄😄😄\n2nd line", 26, "FAILED some::nodeid - ...")
    check("😄😄😄😄😄\n2nd line", 27, "FAILED some::nodeid - 😄...")
    check("😄😄😄😄😄\n2nd line", 28, "FAILED some::nodeid - 😄...")
    check("😄😄😄😄😄\n2nd line", 29, "FAILED some::nodeid - 😄😄...")

    # NOTE: constructed, not sure if this is supported.
    mocked_pos = "nodeid::😄::withunicode"
    check("😄😄😄😄😄\n2nd line", 29, "FAILED nodeid::😄::withunicode")
    check("😄😄😄😄😄\n2nd line", 40, "FAILED nodeid::😄::withunicode - 😄😄...")
    check("😄😄😄😄😄\n2nd line", 41, "FAILED nodeid::😄::withunicode - 😄😄...")
    check("😄😄😄😄😄\n2nd line", 42, "FAILED nodeid::😄::withunicode - 😄😄😄...")
    check("😄😄😄😄😄\n2nd line", 80, "FAILED nodeid::😄::withunicode - 😄😄😄😄😄")


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (10.0, "10.00s"),
        (10.34, "10.34s"),
        (59.99, "59.99s"),
        (60.55, "60.55s (0:01:00)"),
        (123.55, "123.55s (0:02:03)"),
        (60 * 60 + 0.5, "3600.50s (1:00:00)"),
    ],
)
def test_format_session_duration_gate(seconds, expected):
    from _pytest.terminal import format_session_duration

    assert format_session_duration(seconds) == expected

    assert _format_trimmed(" ({}) ", msg, len(msg) + 4) == " (unconditional skip) "
    assert _format_trimmed(" ({}) ", msg, len(msg) + 3) == " (unconditional ...) "

import itertools
class TestMarkerInheritance:
    def test_marker_inheritance_repro(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            '''
            import pytest

            class BaseMeta(type):
                @property
                def pytestmark(self):
                    return (
                        getattr(self, "_pytestmark", []) +
                        list(itertools.chain.from_iterable(getattr(x, "_pytestmark", []) for x in self.__mro__))
                    )

                @pytestmark.setter
                def pytestmark(self, value):
                    # [removed: private attribute assignment — use public API instead]

            class Foo(metaclass=BaseMeta):
                pytestmark = pytest.mark.foo

            class Bar(metaclass=BaseMeta):
                pytestmark = pytest.mark.bar

            class TestDings(Foo, Bar):
                def test_dings_repro(self):
                    pass
            '''
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines([
            '*::test_dings_gate PASSED*',
        ])
        result.assert_outcomes(passed=1)
        assert 'foo' in result.stdout.str()
        assert 'bar' in result.stdout.str()
