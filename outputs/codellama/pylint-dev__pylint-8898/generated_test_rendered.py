# Licensed under the GPL: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# For details: https://github.com/pylint-dev/pylint/blob/main/LICENSE
# Copyright (c) https://github.com/pylint-dev/pylint/blob/main/CONTRIBUTORS.txt

"""Test for the 'pylint-config generate' command."""


import warnings

import pytest
from pytest import CaptureFixture

from pylint import _run_pylint_config


def test_invocation_of_pylint_config_gate(capsys: CaptureFixture[str]) -> None:
    """Check that the help messages are displayed correctly."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="NOTE:.*", category=UserWarning)
        with pytest.raises(SystemExit) as ex:
            _run_pylint_config([""])
        captured = capsys.readouterr()
        assert captured.err.startswith("usage: pylint-config [options]")
        assert ex.value.code == 2

def test_bad_names_rgxs_with_commas_repro(capsys: CaptureFixture[str]) -> None:
    """Check that bad-name-rgxs with commas does not cause a crash."""
    config_content = """
[tool.pylint.basic]
bad-name-rgxs = "(foo{1,3})"
"""
    with open('temp_config.ini', 'w') as f:
        f.write(config_content)
    try:
        _run_pylint_config(['--rcfile=temp_config.ini', 'foo.py'])
    except Exception as exc:
        assert False, f'Unexpected exception raised: {exc}'
    finally:
        import os
        os.remove('temp_config.ini')
