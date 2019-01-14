import pytest
import numpy as np
from disk import *

def test_f1():
    for i in range(2,12):
        assert integrate(f1, i) == pytest.approx(0)

def test_f2():
    for i in range(3,12):
        assert integrate(f2, i) == pytest.approx(np.pi/24)

def test_f3():
    for i in range(1,12):
        assert integrate(f3, i) == pytest.approx(np.pi)

def test_f4():
    for i in range(2,12):
        assert integrate(f4, i) == pytest.approx(np.pi/4)

def test_f5():
    for i in range(2,12):
        assert integrate(f5, i) == pytest.approx(np.pi/4)

def test_f6():
    for i in range(6,12):
        assert integrate(f6,i) == pytest.approx(3.27292)

