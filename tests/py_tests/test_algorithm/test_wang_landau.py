from muca.algorithm.wang_landau import BaseWangLandauResults, WangLandau
from muca.algorithm.parameters import WangLandauParameters
from muca.algorithm.util import OrderParameters
from muca.results.simulation_result import WangLandauResults
from muca.model.p_body_ising import PBodyTwoDimIsing
import numpy as np


def test_base_wang_landau_results():
    base_wang_landau_results = BaseWangLandauResults(
        entropy_dict={1: 2.0, 4: 3.0},
        order_parameters=OrderParameters(
            mag_2={1: 1.0, 4: 3.3},
            abs_f2={1: 2.0, 4: 3.0},
            abs_f4={1: 4.0, 4: 6.0},
            normalized_energy_count={1: 10, 4: 20},
        ),
        total_sweeps=10000,
        final_modification_factor=0.5,
    )

    assert base_wang_landau_results.entropy_dict == {1: 2.0, 4: 3.0}
    assert base_wang_landau_results.order_parameters.mag_2 == {1: 1.0, 4: 3.3}
    assert base_wang_landau_results.order_parameters.abs_f2 == {1: 2.0, 4: 3.0}
    assert base_wang_landau_results.order_parameters.abs_f4 == {1: 4.0, 4: 6.0}
    assert base_wang_landau_results.order_parameters.normalized_energy_count == {1: 10, 4: 20}
    assert base_wang_landau_results.total_sweeps == 10000
    assert base_wang_landau_results.final_modification_factor == 0.5


def test_wang_landau_py():
    model = PBodyTwoDimIsing(J=-1, p=3, Lx=6, Ly=6, spin=0.5, spin_scale_factor=2)
    parameters = WangLandauParameters(
        modification_criterion=1e-01,
        convergence_check_interval=10,
        num_divided_energy_range=2,
        overlap_rate = 0.4,
        seed=0,
        flatness_criterion = 0.9,
    )
    result = WangLandau.run(
        model=model,
        parameters=parameters,
        num_threads=2,
        calculate_order_parameters=True,
        backend = "py"
    )
    ref = WangLandauResults.load_from_pickle("./tests/data/wl_p3_L6_S0.5.pkl")
    assert result.parameters == parameters
    assert result.entropies.size == ref.entropies.size
    assert (result.energies == ref.energies).all()
    assert (result.normalized_energies == ref.normalized_energies).all()
    assert isinstance(result.total_sweeps, int)
    assert result.final_modification_factor <= 1e-01
    assert result.order_parameters.squared_magnetization.size == result.energies.size
    assert result.order_parameters.abs_fourier_second.size == result.energies.size
    assert result.order_parameters.abs_fourier_fourth.size == result.energies.size
    assert (result.order_parameters.energies == ref.order_parameters.energies).all()
    assert (result.order_parameters.normalized_energies == ref.order_parameters.normalized_energies).all()
    assert result.model == model

def test_wang_landau_cpp():
    model = PBodyTwoDimIsing(J=-1, p=3, Lx=6, Ly=6, spin=0.5, spin_scale_factor=2)
    parameters = WangLandauParameters(
        modification_criterion=1e-12,
        convergence_check_interval=100,
        num_divided_energy_range=2,
        overlap_rate = 0.4,
        seed=0,
        flatness_criterion = 0.9,
    )
    result = WangLandau.run(
        model=model,
        parameters=parameters,
        num_threads=2,
        calculate_order_parameters=True,
        backend = "cpp"
    )
    ref = WangLandauResults.load_from_pickle("./tests/data/wl_p3_L6_S0.5.pkl")
    assert result.parameters == parameters
    assert result.entropies.size == ref.entropies.size
    assert (result.energies == ref.energies).all()
    assert (result.normalized_energies == ref.normalized_energies).all()
    assert isinstance(result.total_sweeps, int)
    assert result.final_modification_factor <= 1e-02
    assert result.order_parameters.squared_magnetization.size == result.energies.size
    assert result.order_parameters.abs_fourier_second.size == result.energies.size
    assert result.order_parameters.abs_fourier_fourth.size == result.energies.size
    assert (result.order_parameters.energies == ref.order_parameters.energies).all()
    assert (result.order_parameters.normalized_energies == ref.order_parameters.normalized_energies).all()
    assert result.model == model
    assert abs(result.entropies[0] - np.log(16))/np.log(16) < 0.1