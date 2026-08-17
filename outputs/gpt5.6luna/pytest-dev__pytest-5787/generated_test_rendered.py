import py

failure_demo = py.path.local(__file__).dirpath("failure_demo.py")
pytest_plugins = ("pytester",)


def test_failure_demo_fails_properly_gate(testdir):
    target = testdir.tmpdir.join(failure_demo.basename)
    failure_demo.copy(target)
    failure_demo.copy(testdir.tmpdir.join(failure_demo.basename))
    result = testdir.runpytest(target, syspathinsert=True)
    result.stdout.fnmatch_lines(["*44 failed*"])
    assert result.ret != 0

def test_chained_exception_with_from_repro(testdir):
    test_content = '''
def test_chained_exception_with_from_repro():
    try:
        try:
            raise ValueError(11)
        except Exception as e1:
            raise ValueError(12) from e1
    except Exception as e2:
        raise ValueError(13) from e2
'''
    testdir.makepyfile(test_content)
    result = testdir.runpytest()
    assert result.ret == 1
