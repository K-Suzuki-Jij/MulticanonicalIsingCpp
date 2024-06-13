from muca.algorithm.multicanonical import BaseMulticanonicalResults, Multicanonical
from muca.algorithm.util import OrderParameters
from muca.algorithm.parameters import MulticanonicalParameters
from muca.results.analysis import WangLandauResults
import numpy as np

def test_base_multicanonical_results():
    base_multicanonical_results = BaseMulticanonicalResults(
        histogram_dict={1: 2, 4: 3},
        order_parameters=OrderParameters(
            sq_mag={1: 1.0, 4: 3.3},
            abs_f2={1: 2.0, 4: 3.0},
            abs_f4={1: 4.0, 4: 6.0},
            normalized_energy_count={1: 10, 4: 20},
        ),
    )

    assert base_multicanonical_results.histogram_dict == {1: 2.0, 4: 3.0}
    assert base_multicanonical_results.order_parameters.sq_mag == {1: 1.0, 4: 3.3}
    assert base_multicanonical_results.order_parameters.abs_f2 == {1: 2.0, 4: 3.0}
    assert base_multicanonical_results.order_parameters.abs_f4 == {1: 4.0, 4: 6.0}
    assert base_multicanonical_results.order_parameters.normalized_energy_count == {1: 10, 4: 20}

def test_multicanonical_py():
    parameters = MulticanonicalParameters(
        num_sweeps=10000,
        num_divided_energy_range=2,
        overlap_rate=0.6,
        seed=0,
    )
    wl_results = WangLandauResults.load_from_pickle("./tests/data/wl_p3_L6_S0.5.pkl")
    
    muca_results = Multicanonical.multicanonical(
        initial_data=wl_results,
        parameters=parameters,
        num_threads=2,
        calculate_order_parameters=True,
        backend="py",
    )
    assert muca_results.initial_data == wl_results
    assert muca_results.parameters == parameters
    assert muca_results.entropies.size == wl_results.energies.size
    assert (muca_results.energies == wl_results.energies).all()
    assert (muca_results.normalized_energies == wl_results.normalized_energies).all()
    assert muca_results.histogram.size == wl_results.energies.size
    assert (muca_results.order_parameters.energies == wl_results.order_parameters.energies).all()
    assert (muca_results.order_parameters.normalized_energies == wl_results.order_parameters.normalized_energies).all()
    assert muca_results.order_parameters.squared_magnetization.size == muca_results.energies.size
    assert muca_results.order_parameters.abs_fourier_second.size == muca_results.energies.size
    assert muca_results.order_parameters.abs_fourier_fourth.size == muca_results.energies.size
    assert muca_results.model == wl_results.model
    assert abs(muca_results.entropies[0] - np.log(16))/np.log(16) < 0.1


def test_multicanonical_cpp():
    parameters = MulticanonicalParameters(
        num_sweeps=10000,
        num_divided_energy_range=2,
        overlap_rate=0.6,
        seed=3,
    )
    wl_results = WangLandauResults.load_from_pickle("./tests/data/wl_p3_L6_S0.5.pkl")
    
    muca_results = Multicanonical.multicanonical(
        initial_data=wl_results,
        parameters=parameters,
        num_threads=2,
        calculate_order_parameters=True,
        backend="cpp",
    )
    assert muca_results.initial_data == wl_results
    assert muca_results.parameters == parameters
    assert muca_results.entropies.size == wl_results.energies.size
    assert (muca_results.energies == wl_results.energies).all()
    assert (muca_results.normalized_energies == wl_results.normalized_energies).all()
    assert muca_results.histogram.size == wl_results.energies.size
    assert (muca_results.order_parameters.energies == wl_results.order_parameters.energies).all()
    assert (muca_results.order_parameters.normalized_energies == wl_results.order_parameters.normalized_energies).all()
    assert muca_results.order_parameters.squared_magnetization.size == muca_results.energies.size
    assert muca_results.order_parameters.abs_fourier_second.size == muca_results.energies.size
    assert muca_results.order_parameters.abs_fourier_fourth.size == muca_results.energies.size
    assert muca_results.model == wl_results.model
    assert abs(muca_results.entropies[0] - np.log(16))/np.log(16) < 0.1