from muca.algorithm.util import OrderParameters, OrderParameterCounter, AlgorithmUtil
from muca.model.p_body_ising import PBodyTwoDimIsing
import numpy as np





def test_algorithm_util_generate_initial_state():
    model = PBodyTwoDimIsing(J=-1, p=2, Lx=4, Ly=4, spin=0.5, spin_scale_factor=2)
    initial_state = AlgorithmUtil.generate_initial_state(model, (-32.0, -31.0))
    assert np.sum(initial_state) == 16 or np.sum(initial_state) == -16