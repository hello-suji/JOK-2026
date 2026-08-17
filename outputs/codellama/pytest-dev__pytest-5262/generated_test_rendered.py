def test_foo_gate():
    pass

import pytest
from _pytest.capture import EncodedFile

def test_encodedfile_write_bytes_repro():
    ef = EncodedFile(open('test_output.txt', 'wb'), encoding='utf-8')
    ef.write(b'test bytes')
    ef.close()
    with open('test_output.txt', 'r', encoding='utf-8') as f:
        assert f.read() == 'test bytes'
