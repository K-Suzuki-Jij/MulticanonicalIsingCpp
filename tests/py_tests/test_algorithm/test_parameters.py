from muca.algorithm.parameters import WangLandauParameters, MulticanonicalParameters


def test_WangLandauParameters():
    wang_landau_parameters = WangLandauParameters(
        modification_criterion=1e-8,
        convergence_check_interval=1000,
        num_divided_energy_range=3,
        seed=10,
        max_sweeps=99999,
        flatness_criterion=0.7,
        reduce_rate=0.3,
        overlap_rate=0.6
    )

    assert wang_landau_parameters.modification_criterion == 1e-8
    assert wang_landau_parameters.convergence_check_interval == 1000
    assert wang_landau_parameters.num_divided_energy_range == 3
    assert wang_landau_parameters.seed == 10
    assert wang_landau_parameters.max_sweeps == 99999
    assert wang_landau_parameters.flatness_criterion == 0.7
    assert wang_landau_parameters.reduce_rate == 0.3
    assert wang_landau_parameters.overlap_rate == 0.6


def test_MulticanonicalParameters():
    multicanonical_parameters = MulticanonicalParameters(
        num_sweeps=10000,
        num_divided_energy_range=2,
        overlap_rate=0.6,
        seed=0,
    )

    assert multicanonical_parameters.num_sweeps == 10000
    assert multicanonical_parameters.num_divided_energy_range == 2
    assert multicanonical_parameters.overlap_rate == 0.6
    assert multicanonical_parameters.seed == 0