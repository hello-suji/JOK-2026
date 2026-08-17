import os.path
import shutil

failure_demo = os.path.join(os.path.dirname(__file__), "failure_demo.py")
pytest_plugins = ("pytester",)


def test_failure_demo_fails_properly(testdir):
    target = testdir.tmpdir.join(os.path.basename(failure_demo))
    shutil.copy(failure_demo, target)
    result = testdir.runpytest(target, syspathinsert=True)
    result.stdout.fnmatch_lines(["*44 failed*"])
    assert result.ret != 0

import pytest
from _pytest.skipping import pytest_runtest_makereport

def test_pytest_runtest_makereport_skip_location_repro(testdir):
    testdir.makepyfile('''
import pytest

@pytest.mark.skip
def test_example_repro():
    assert 0
''')
    result = testdir.runpytest()
    report = pytest_runtest_makereport(result)
    assert 'src/_pytest/skipping.py' not in report.longreprtext
    # [Tier 2: probe-verified buggy repr — must differ after fix]
    assert repr(report) != 'SKIPPED [1] src/_pytest/skipping.py:238: unconditional skip'
