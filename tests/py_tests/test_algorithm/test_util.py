from muca.algorithm.util import OrderParameters, OrderParameterCounter, AlgorithmUtil
from muca.model.p_body_ising import PBodyTwoDimIsing
import numpy as np
import math




def test_algorithm_util_generate_initial_state():
    model = PBodyTwoDimIsing(J=-1, p=2, Lx=4, Ly=4, spin=0.5, spin_scale_factor=2)
    initial_state = AlgorithmUtil.generate_initial_state(model, (-32.0, -31.0))
    assert np.sum(initial_state) == 16 or np.sum(initial_state) == -16

def test_algorithm_util_generate_energy_range_list():
    r = AlgorithmUtil.generate_energy_range_list([-10, 10], 2, 0)
    assert r[0][0] == -10
    assert r[0][1] == 0
    assert r[1][0] == 0
    assert r[1][1] == 10

    r = AlgorithmUtil.generate_energy_range_list([-10, 10], 3, 0)
    assert r[0][0] == -10
    assert math.isclose(r[0][1], -10/3)
    assert math.isclose(r[1][0], -10/3)
    assert math.isclose(r[1][1], 10/3)
    assert math.isclose(r[2][0], 10/3)
    assert r[2][1] == 10

    r = AlgorithmUtil.generate_energy_range_list([-10, 10], 2, 0.4)
    assert r[0][0] == -10
    assert math.isclose(r[0][1], +2.5)
    assert math.isclose(r[1][0], -2.5)
    assert r[1][1] == 10

    r = AlgorithmUtil.generate_energy_range_list([-10, 10], 3, 0.4)
    assert r[0][0] == -10
    assert math.isclose(r[0][1], -10 + 20/2.2)
    assert math.isclose(r[1][0], -10 + 0.6*20/2.2)
    assert math.isclose(r[1][1], -10 + 1.6*20/2.2)
    assert math.isclose(r[2][0], -10 + 120.0/11)
    assert r[2][1] == 10

    r = AlgorithmUtil.generate_energy_range_list([-11, 11], 7, 0.4)
    assert r[0][0] == -11
    assert r[6][1] == +11