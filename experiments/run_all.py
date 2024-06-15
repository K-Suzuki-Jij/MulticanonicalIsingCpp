from muca.algorithm.multicanonical import Multicanonical
from muca.algorithm.parameters import MulticanonicalParameters, WangLandauParameters
from muca.algorithm.wang_landau import WangLandau
from muca.model.p_body_ising import PBodyTwoDimIsing

p = 3
L = 30
S = 1 / 2
num_divided_energy_range = 1
modification_criterion = 1e-12
flatness_criterion = 0.9
overlap_rate = 0.5
model = PBodyTwoDimIsing(J=-1, p=p, Lx=L, Ly=L, spin=S, spin_scale_factor=1 / S)
wl_parameters = WangLandauParameters(
    modification_criterion=modification_criterion,
    convergence_check_interval=10000,
    num_divided_energy_range=num_divided_energy_range,
    overlap_rate=overlap_rate,
    flatness_criterion=flatness_criterion,
)

wl_results = WangLandau.run(
    model=model,
    parameters=wl_parameters,
    num_threads=num_divided_energy_range,
    calculate_order_parameters=True,
    backend="cpp",
)
wl_results.store_as_pickle(
    f"wl_p{p}_L{L}_S{S}_{modification_criterion}_{flatness_criterion}_{overlap_rate}.pkl"
)


muca_sweeps = 10 ** (len(str(wl_results.total_sweeps)))
muca_parameters = MulticanonicalParameters(
    num_sweeps=muca_sweeps,
    num_divided_energy_range=wl_results.parameters.num_divided_energy_range,
    overlap_rate=wl_results.parameters.overlap_rate,
)

muca_result = Multicanonical.multicanonical(
    initial_data=wl_results,
    parameters=muca_parameters,
    num_threads=wl_results.parameters.num_divided_energy_range,
    calculate_order_parameters=True,
    backend="cpp",
)

muca_result.store_as_pickle(
    f"muca_p{p}_L{L}_S{S}_{modification_criterion}_{flatness_criterion}_{overlap_rate}.pkl"
)
