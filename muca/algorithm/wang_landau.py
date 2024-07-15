import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import reduce
from itertools import product, repeat
from typing import Optional

import numpy as np

from muca import cpp_muca
from muca.algorithm.parameters import WangLandauParameters, UpdateMethod
from muca.algorithm.util import AlgorithmUtil, OrderParameterCounter, OrderParameters
from muca.model.caster import cast_from_py_model, cast_from_py_wang_landau_parameters
from muca.model.p_body_ising import PBodyTwoDimIsing
from muca.results.simulation_result import OrderParameterResults, WangLandauResults


@dataclass
class BaseWangLandauResults:
    """Class to store the results of the Wang-Landau simulation.

    Attributes:
        entropy_dict (Optional[dict[int, float]]): The entropy of the system.
        order_parameters (Optional[OrderParameters]): The order parameters of the system.
        total_sweeps (Optional[int]): The total number of sweeps.
        final_modification_factor (Optional[float]): The final modification factor.
    """

    entropy_dict: Optional[dict[int, float]] = None
    order_parameters: Optional[OrderParameters] = None
    total_sweeps: Optional[int] = None
    final_modification_factor: Optional[float] = None


def _base_wang_landau(
    model: PBodyTwoDimIsing,
    parameters: WangLandauParameters,
    normalized_energy_range: tuple[float, float],
    calculate_order_parameters: bool,
) -> BaseWangLandauResults:
    """Run a Wang-Landau simulation.

    Args:
        model (PBodyTwoDimIsing): The model.
        parameters (WangLandauParameters): The parameters for the simulation.
        normalized_energy_range (tuple[float, float]): The range of normalized energies to consider.
        calculate_order_parameters (bool): Whether to calculate the order parameters.

    Returns:
        BaseWangLandauResults: The results of the simulation.
    """
    e_min, e_max = normalized_energy_range
    twice_spins = AlgorithmUtil.generate_initial_state(model, normalized_energy_range)
    dE = model.make_energy_difference(twice_spins)
    histogram_dict: dict[int, int] = defaultdict(int)
    entropy_dict: dict[int, float] = defaultdict(float)
    diff: float = 1.0
    loop_breaker: bool = False
    normalized_energy: int = model.calculate_normalized_energy(twice_spins)
    order_parameter_counter = OrderParameterCounter(model, twice_spins)

    for sweep in range(parameters.max_sweeps):
        if loop_breaker:
            break

        for x, y in product(range(model.Lx), range(model.Ly)):
            new_spin_value = np.random.choice(
                model.twice_spin_set[model.twice_spin_set != twice_spins[x, y]]
            )
            new_normalized_energy = normalized_energy + dE[x, y] * (
                new_spin_value - twice_spins[x, y]
            )
            dS = entropy_dict[new_normalized_energy] - entropy_dict[normalized_energy]
            if (e_min <= new_normalized_energy <= e_max) and (
                dS <= 0 or np.random.rand() < np.exp(-dS)
            ):
                if calculate_order_parameters:
                    order_parameter_counter.update_fourier(
                        new_spin_value - twice_spins[x, y], x, y
                    )
                model.update_energy_difference(new_spin_value, x, y, dE, twice_spins)
                twice_spins[x, y] = new_spin_value
                normalized_energy = new_normalized_energy
            entropy_dict[normalized_energy] += diff
            histogram_dict[normalized_energy] += 1

            if calculate_order_parameters:
                order_parameter_counter.update_order_parameters(normalized_energy)

        # Check if the histogram is flat
        if sweep % parameters.convergence_check_interval == 0:
            hist_mean = np.mean(list(histogram_dict.values()))
            hist_min = np.min(list(histogram_dict.values()))
            if hist_min > hist_mean * parameters.flatness_criterion:
                if diff < parameters.modification_criterion:
                    loop_breaker = True
                diff *= parameters.reduce_rate
                histogram_dict.clear()

    return BaseWangLandauResults(
        entropy_dict=dict(sorted(entropy_dict.items())),
        order_parameters=order_parameter_counter.to_order_parameters(),
        total_sweeps=sweep + 1,
        final_modification_factor=diff / parameters.reduce_rate,
    )


def _post_process(
    model: PBodyTwoDimIsing,
    result_list: list[BaseWangLandauResults],
) -> tuple[np.ndarray, np.ndarray, OrderParameters, float]:
    """Post-process the results of the Wang-Landau simulation.

    Args:
        model (PBodyTwoDimIsing): The model.
        result_list (list[BaseWangLandauResults]): The results of the Wang-Landau simulation.

    Returns:
        tuple[np.ndarray, np.ndarray, OrderParameterResults, float]:
            The entropies, normalized energies, order parameters, and energy coefficient.
    """
    # Merge the results
    entropy_dict, order_parameters = AlgorithmUtil.merge_results(
        normalized_energy_range=model.normalized_energy_range,
        separated_data_list=[r.entropy_dict for r in result_list],
        order_parameters_list=[r.order_parameters for r in result_list],
        connect_edge=True,
    )
    energy_coeff = abs(model.J) * (model.spin_scale_factor / 2) ** model.p

    # Sort
    normalized_energies, entropies = zip(
        *sorted(entropy_dict.items(), key=lambda x: x[0] * energy_coeff)
    )

    # Scale entropies
    entropies = AlgorithmUtil.scale_entropies(entropies, model)

    return (
        np.array(entropies),
        np.array(normalized_energies),
        order_parameters,
        energy_coeff,
    )


class WangLandau:

    @staticmethod
    def wang_landau_py(
        model: PBodyTwoDimIsing,
        parameters: WangLandauParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
    ) -> WangLandauResults:
        """Run a Wang-Landau simulation.

        Args:
            model (PBodyTwoDimIsing): The model.
            parameters (WangLandauParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.

        Returns:
            WangLandauResults: The results of the simulation.
        """
        start_wang_landau = time.perf_counter()

        # Get energy range
        normalized_energy_range_list = AlgorithmUtil.generate_energy_range_list(
            normalized_energy_range=model.normalized_energy_range,
            num_divided_energy_range=parameters.num_divided_energy_range,
            overlap_rate=parameters.overlap_rate,
        )

        # Run the simulation
        print("Running Wang-Landau simulation by python ...", flush=True)
        start_simulation = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            result_list = list(
                executor.map(
                    _base_wang_landau,
                    repeat(model),
                    repeat(parameters),
                    normalized_energy_range_list,
                    repeat(calculate_order_parameters),
                )
            )
        end_simulation = time.perf_counter()
        print(
            f"Done simulation ({round(end_simulation - start_simulation, 1)}) [sec] by python",
            flush=True,
        )

        (
            entropies,
            normalized_energies,
            order_parameters,
            energy_coeff,
        ) = _post_process(model, result_list)

        return WangLandauResults(
            parameters=parameters,
            entropies=entropies,
            energies=normalized_energies * energy_coeff,
            normalized_energies=normalized_energies,
            total_sweeps=sum([r.total_sweeps for r in result_list]),
            final_modification_factor=result_list[0].final_modification_factor,
            order_parameters=order_parameters.to_order_parameter_results(energy_coeff),
            model=model,
            info={
                "simulation_time": end_simulation - start_simulation,
                "total_time": time.perf_counter() - start_wang_landau,
                "backend": "py",
                "num_threads": num_threads,
                "calculate_order_parameters": calculate_order_parameters,
                "order_parameters": order_parameters,
            },
        )

    @staticmethod
    def wang_landau_cpp(
        model: PBodyTwoDimIsing,
        parameters: WangLandauParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
    ) -> WangLandauResults:
        """Run a Wang-Landau simulation.

        Args:
            model (PBodyTwoDimIsing): The model.
            parameters (WangLandauParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.

        Returns:
            WangLandauResults: The results of the simulation.
        """
        start_wang_landau = time.perf_counter()

        # Run the simulation
        print("Running Wang-Landau simulation by cpp ...", flush=True)
        start_simulation = time.perf_counter()
        result_list = cpp_muca.cpp_algorithm.run_wang_landau(
            model=cast_from_py_model(model),
            parameters=cast_from_py_wang_landau_parameters(parameters),
            num_threads=num_threads,
            calculate_order_parameters=calculate_order_parameters,
        )
        end_simulation = time.perf_counter()
        print(
            f"Done simulation ({round(end_simulation - start_simulation, 1)}) [sec] by cpp",
            flush=True,
        )

        (
            entropies,
            normalized_energies,
            order_parameters,
            energy_coeff,
        ) = _post_process(model, result_list)

        return WangLandauResults(
            parameters=parameters,
            entropies=entropies,
            energies=normalized_energies * energy_coeff,
            normalized_energies=normalized_energies,
            total_sweeps=sum([r.total_sweeps for r in result_list]),
            final_modification_factor=result_list[0].final_modification_factor,
            order_parameters=order_parameters.to_order_parameter_results(energy_coeff),
            model=model,
            info={
                "simulation_time": end_simulation - start_simulation,
                "total_time": time.perf_counter() - start_wang_landau,
                "backend": "cpp",
                "num_threads": num_threads,
                "calculate_order_parameters": calculate_order_parameters,
                "order_parameters": order_parameters,
            },
        )
    
    @staticmethod
    def wang_landau_cpp_symmetric(
        model: PBodyTwoDimIsing,
        parameters: WangLandauParameters,
    ) -> WangLandauResults:
        """Run a Wang-Landau simulation with using symmetry of the system.

        Args:
            model (PBodyTwoDimIsing): The model.
            parameters (WangLandauParameters): The parameters for the simulation.

        Returns:
            WangLandauResults: The results of the simulation.
        """
        start_wang_landau = time.perf_counter()

        # Check if the parameters are valid
        if parameters.num_divided_energy_range != 1:
            raise ValueError(
                "The number of divided energy range must be 1 for symmetric Wang-Landau simulation."
            )
        if parameters.update_method not in [UpdateMethod.METROPOLIS, "METROPOLIS"]:
            raise ValueError(
                "The update method must be Metropolis for symmetric Wang-Landau simulation."
            )

        # Run the simulation
        print("Running symmetric Wang-Landau simulation by cpp ...", flush=True)
        start_simulation = time.perf_counter()
        result = cpp_muca.cpp_algorithm.run_wang_landau_symmetric(
            model=cast_from_py_model(model),
            parameters=cast_from_py_wang_landau_parameters(parameters),
        )
        end_simulation = time.perf_counter()
        print(
            f"Done simulation ({round(end_simulation - start_simulation, 1)}) [sec] by cpp",
            flush=True,
        )

        (
            entropies,
            normalized_energies,
            order_parameters,
            energy_coeff,
        ) = _post_process(model, [result])

        return WangLandauResults(
            parameters=parameters,
            entropies=entropies,
            energies=normalized_energies * energy_coeff,
            normalized_energies=normalized_energies,
            total_sweeps=result.total_sweeps,
            final_modification_factor=result.final_modification_factor,
            order_parameters=order_parameters.to_order_parameter_results(energy_coeff),
            model=model,
            info={
                "simulation_time": end_simulation - start_simulation,
                "total_time": time.perf_counter() - start_wang_landau,
                "backend": "cpp",
                "num_threads": 1,
                "calculate_order_parameters": False,
                "order_parameters": order_parameters,
            },
        )

    @classmethod
    def run(
        cls,
        model: PBodyTwoDimIsing,
        parameters: WangLandauParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
        backend: str = "py",
        symmetric_calculation: bool = False,
    ) -> WangLandauResults:
        """Run a Wang-Landau simulation.

        Args:
            model (PBodyTwoDimIsing): The model.
            parameters (WangLandauParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.
            backend (str, optional): The backend to use. "py" or "cpp". Defaults to "py".

        Returns:
            WangLandauResults: The results of the simulation.
        """
        if symmetric_calculation:
            if backend == "py":
                raise ValueError("Symmetric calculation is only supported by cpp backend.")
            if num_threads != 1:
                raise ValueError("Symmetric calculation is only supported by single thread.")
            if calculate_order_parameters:
                raise ValueError("Symmetric calculation is only supported without order parameters.")
            return cls.wang_landau_cpp_symmetric(model=model, parameters=parameters)
        else:
            if backend == "py":
                return cls.wang_landau_py(
                    model=model,
                    parameters=parameters,
                    num_threads=num_threads,
                    calculate_order_parameters=calculate_order_parameters,
                )
            elif backend == "cpp":
                return cls.wang_landau_cpp(
                    model=model,
                    parameters=parameters,
                    num_threads=num_threads,
                    calculate_order_parameters=calculate_order_parameters,
                )
            else:
                raise ValueError(f"Invalid backend: {backend}")
