from muca import cpp_muca
from muca.algorithm.parameters import MulticanonicalParameters, WangLandauParameters
from muca.model.p_body_ising import PBodyTwoDimIsing


def cast_from_py_model(model: PBodyTwoDimIsing) -> cpp_muca.cpp_model.PBodyTwoDimIsing:
    return cpp_muca.cpp_model.PBodyTwoDimIsing(
        J=model.J,
        p=model.p,
        Lx=model.Lx,
        Ly=model.Ly,
        spin=model.spin,
        spin_scale_factor=model.spin_scale_factor,
    )


def cast_from_py_wang_landau_parameters(
    parameters: WangLandauParameters,
) -> cpp_muca.cpp_algorithm.WangLandauParameters:
    return cpp_muca.cpp_algorithm.WangLandauParameters(
        modification_criterion=parameters.modification_criterion,
        convergence_check_interval=parameters.convergence_check_interval,
        num_divided_energy_range=parameters.num_divided_energy_range,
        seed=parameters.seed,
        max_sweeps=parameters.max_sweeps,
        flatness_criterion=parameters.flatness_criterion,
        reduce_rate=parameters.reduce_rate,
        overlap_rate=parameters.overlap_rate,
    )


def cast_from_py_multicanonical_parameters(
    parameters: MulticanonicalParameters,
) -> cpp_muca.cpp_algorithm.MulticanonicalParameters:
    return cpp_muca.cpp_algorithm.MulticanonicalParameters(
        num_sweeps=parameters.num_sweeps,
        num_divided_energy_range=parameters.num_divided_energy_range,
        overlap_rate=parameters.overlap_rate,
        seed=parameters.seed,
    )
