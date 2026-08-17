from _pytest.pytester import Pytester


def test_no_items_should_not_show_output_gate(pytester: Pytester) -> None:
    result = pytester.runpytest("--fixtures-per-test")
    result.stdout.no_fnmatch_line("*fixtures used by*")
    assert result.ret == 0


def test_fixtures_in_module_gate(pytester: Pytester) -> None:
    p = pytester.makepyfile(
        '''
        import pytest
        @pytest.fixture
        def _arg0():
            """hidden arg0 fixture"""
        @pytest.fixture
        def arg1():
            """arg1 docstring"""
        def test_arg1_gate(arg1):
            pass
    '''
    )

    result = pytester.runpytest("--fixtures-per-test", p)
    assert result.ret == 0

    result.stdout.fnmatch_lines(
        [
            "*fixtures used by test_arg1_gate*",
            "*(test_fixtures_in_module_gate.py:9)*",
            "arg1",
            "    arg1 docstring",
        ]
    )
    result.stdout.no_fnmatch_line("*_arg0*")


def test_fixtures_in_conftest_gate(pytester: Pytester) -> None:
    pytester.makeconftest(
        '''
        import pytest
        @pytest.fixture
        def arg1():
            """arg1 docstring"""
        @pytest.fixture
        def arg2():
            """arg2 docstring"""
        @pytest.fixture
        def arg3(arg1, arg2):
            """arg3
            docstring
            """
    '''
    )
    p = pytester.makepyfile(
        """
        def test_arg2_gate(arg2):
            pass
        def test_arg3_gate(arg3):
            pass
    """
    )
    result = pytester.runpytest("--fixtures-per-test", p)
    assert result.ret == 0

    result.stdout.fnmatch_lines(
        [
            "*fixtures used by test_arg2_gate*",
            "*(test_fixtures_in_conftest_gate.py:2)*",
            "arg2",
            "    arg2 docstring",
            "*fixtures used by test_arg3_gate*",
            "*(test_fixtures_in_conftest_gate.py:4)*",
            "arg1",
            "    arg1 docstring",
            "arg2",
            "    arg2 docstring",
            "arg3",
            "    arg3",
            "    docstring",
        ]
    )


def test_should_show_fixtures_used_by_test_gate(pytester: Pytester) -> None:
    pytester.makeconftest(
        '''
        import pytest
        @pytest.fixture
        def arg1():
            """arg1 from conftest"""
        @pytest.fixture
        def arg2():
            """arg2 from conftest"""
    '''
    )
    p = pytester.makepyfile(
        '''
        import pytest
        @pytest.fixture
        def arg1():
            """arg1 from testmodule"""
        def test_args_gate(arg1, arg2):
            pass
    '''
    )
    result = pytester.runpytest("--fixtures-per-test", p)
    assert result.ret == 0

    result.stdout.fnmatch_lines(
        [
            "*fixtures used by test_args_gate*",
            "*(test_should_show_fixtures_used_by_test_gate.py:6)*",
            "arg1",
            "    arg1 from testmodule",
            "arg2",
            "    arg2 from conftest",
        ]
    )


def test_verbose_include_private_fixtures_and_loc_gate(pytester: Pytester) -> None:
    pytester.makeconftest(
        '''
        import pytest
        @pytest.fixture
        def _arg1():
            """_arg1 from conftest"""
        @pytest.fixture
        def arg2(_arg1):
            """arg2 from conftest"""
    '''
    )
    p = pytester.makepyfile(
        '''
        import pytest
        @pytest.fixture
        def arg3():
            """arg3 from testmodule"""
        def test_args_gate(arg2, arg3):
            pass
    '''
    )
    result = pytester.runpytest("--fixtures-per-test", "-v", p)
    assert result.ret == 0

    result.stdout.fnmatch_lines(
        [
            "*fixtures used by test_args_gate*",
            "*(test_verbose_include_private_fixtures_and_loc_gate.py:6)*",
            "_arg1 -- conftest.py:3",
            "    _arg1 from conftest",
            "arg2 -- conftest.py:6",
            "    arg2 from conftest",
            "arg3 -- test_verbose_include_private_fixtures_and_loc_gate.py:3",
            "    arg3 from testmodule",
        ]
    )


def test_doctest_items_gate(pytester: Pytester) -> None:
    pytester.makepyfile(
        '''
        def foo():
            """
            >>> 1 + 1
            2
            """
    '''
    )
    pytester.maketxtfile(
        """
        >>> 1 + 1
        2
    """
    )
    assert result.ret == 0

    result.stdout.fnmatch_lines(["*collected 2 items*"])

def test_bytes_warning_in_show_fixture_action_repro(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        @pytest.mark.parametrize('data', [b'Hello World'])
        def test_data_repro(data):
            pass
        """
    )
    result = pytester.runpytest('-bb', '--setup-show')
    assert result.ret == 0
    # [Tier 2: probe-verified buggy repr — must differ after fix]
    assert repr(result) != 'BytesWarning: str() on a bytes instance'
    assert result.ret == 0

    result.stdout.fnmatch_lines(["*collected 2 items*"])
