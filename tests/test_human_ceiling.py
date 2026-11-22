import pytest
from m3.audit.human_ceiling import loo
def test_loo():
    assert loo([10,0,0,0,0,0,0,0])==pytest.approx(1)
    assert loo([5,5,0,0,0,0,0,0])==pytest.approx(0)
    assert loo([6,4,0,0,0,0,0,0])==pytest.approx(0.6)
    assert loo([4,3,3,0,0,0,0,0])==pytest.approx(0.4/3)
