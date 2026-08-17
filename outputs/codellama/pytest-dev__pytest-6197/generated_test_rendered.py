import pytest


@pytest.fixture(scope="session")
def setup(request):
    setup = CostlySetup()
    yield setup
    setup.finalize()


class CostlySetup:
    def __init__(self):
        import time

        print("performing costly setup")
        time.sleep(5)
        self.timecostly = 1

    def finalize(self):
        del self.timecostly

def test_issue_6197_repro(pytester):
    foobar_dir = pytester.mkpydir('foobar')
    foobar_init = foobar_dir.join('__init__.py')
    foobar_init.write('assert False')

    result = pytester.runpytest()
    assert result.ret == 0
