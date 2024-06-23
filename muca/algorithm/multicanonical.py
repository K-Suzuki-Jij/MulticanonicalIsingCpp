import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product, repeat
from typing import Optional

import numpy as np

from muca import cpp_muca
from muca.algorithm.parameters import MulticanonicalParameters
from muca.algorithm.util import AlgorithmUtil, OrderParameterCounter, OrderParameters
from muca.model.caster import cast_from_py_model, cast_from_py_multicanonical_parameters
from muca.model.p_body_ising import PBodyTwoDimIsing
from muca.results.simulation_result import (
    MulticanonicalResults,
    OrderParameterResults,
    WangLandauResults,
)


@dataclass
class BaseMulticanonicalResults:
    """Class to store the results of the multicanonical simulation.

    Attributes:
        histogram_dict (Optional[dict[int, int]]): The histogram of the system.
        order_parameters (Optional[OrderParameters]): The order parameters of the system.
    """

    histogram_dict: Optional[dict[int, int]] = None
    order_parameters: Optional[OrderParameters] = None


def _base_multicanonical(
    model: PBodyTwoDimIsing,
    num_sweeps: int,
    entropy_dict: dict[int, float],
    normalized_energy_range: tuple[float, float],
    calculate_order_parameters: bool,
) -> BaseMulticanonicalResults:
    """Run a multicanonical simulation.

    Args:
        model (PBodyTwoDimIsing): The model.
        num_sweeps (int): The number of sweeps to run.
        entropy_dict (dict[int, float]): The entropy of the system, which is obtained from primary simulation.
        normalized_energy_range (tuple[float, float]): The range of normalized energies to consider.
        calculate_order_parameters (bool): Whether to calculate the order parameters.

    Returns:
        BaseMulticanonicalResults: The results of the simulation.
    """
    e_min, e_max = normalized_energy_range
    twice_spins = AlgorithmUtil.generate_initial_state(model, normalized_energy_range)
    dE = model.make_energy_difference(twice_spins)
    histogram_dict: dict[int, int] = {e: 0 for e in sorted(entropy_dict.keys())}
    normalized_energy: int = model.calculate_normalized_energy(twice_spins)
    order_parameter_counter = OrderParameterCounter(model, twice_spins)

    for _ in range(num_sweeps):
        for x, y in product(range(model.Lx), range(model.Ly)):
            new_spin_value = np.random.choice(
                model.twice_spin_set[model.twice_spin_set != twice_spins[x, y]]
            )
            new_normalized_energy = normalized_energy + dE[x, y] * (
                new_spin_value - twice_spins[x, y]
            )
            if e_min <= new_normalized_energy <= e_max:
                dS = (
                    entropy_dict[new_normalized_energy]
                    - entropy_dict[normalized_energy]
                )
                if dS <= 0 or np.random.rand() < np.exp(-dS):
                    if calculate_order_parameters:
                        order_parameter_counter.update_fourier(
                            new_spin_value - twice_spins[x, y], x, y
                        )
                    model.update_energy_difference(
                        new_spin_value, x, y, dE, twice_spins
                    )
                    twice_spins[x, y] = new_spin_value
                    normalized_energy = new_normalized_energy
            histogram_dict[normalized_energy] += 1

            if calculate_order_parameters:
                order_parameter_counter.update_order_parameters(normalized_energy)

    return BaseMulticanonicalResults(
        histogram_dict=histogram_dict,
        order_parameters=order_parameter_counter.to_order_parameters(),
    )


def _post_process(
    initial_data: WangLandauResults, result_list: list[BaseMulticanonicalResults]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OrderParameters, float]:
    """Post-process the results of the multicanonical simulation.
    Merge the divided results and update the entropies.

    Args:
        initial_data (WangLandauResults): The results of the primary simulation.
        result_list (list[BaseMulticanonicalResults]): The results of the multicanonical simulation.

    Raises:
        ValueError: If the energies obtained in the multicanonical simulation is not same as the initial entropy.
        ValueError: If the histogram has non-positive values.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            The updated entropies, normalized energies, histograms, order parameters, and energy coefficient.
    """
    # Merge the results
    histogram_dict, order_parameters = AlgorithmUtil.merge_results(
        normalized_energy_range=initial_data.model.normalized_energy_range,
        separated_data_list=[r.histogram_dict for r in result_list],
        order_parameters_list=[r.order_parameters for r in result_list],
        connect_edge=True,
    )

    # Merge order parameters
    energy_coeff = (
        abs(initial_data.model.J)
        * (initial_data.model.spin_scale_factor / 2) ** initial_data.model.p
    )

    # Check if the energies are valid
    if set(histogram_dict.keys()) != set(initial_data.normalized_energies):
        raise ValueError("The energy histogram is not same as the initial entropy.")

    # Check if the histogram is valid
    if any(x <= 0 for x in histogram_dict.values()):
        raise ValueError("The histogram has non-positive values.")

    # Sort
    normalized_energies, histograms = zip(
        *sorted(histogram_dict.items(), key=lambda x: x[0] * energy_coeff)
    )

    # Update entropies
    modified_entropies = initial_data.entropies + np.log(histograms)

    # Scale entropies
    modified_entropies = AlgorithmUtil.scale_entropies(
        modified_entropies, initial_data.model
    )

    return (
        np.array(modified_entropies),
        np.array(normalized_energies),
        np.array(histograms),
        order_parameters,
        energy_coeff,
    )


class Multicanonical:
    """Class to run a multicanonical simulation."""

    @staticmethod
    def multicanonical_py(
        initial_data: WangLandauResults,
        parameters: MulticanonicalParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
    ) -> MulticanonicalResults:
        """Run a multicanonical simulation by python implementation.

        Args:
            initial_data (WangLandauResults): The results of the primary simulation.
            parameters (MulticanonicalParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.

        Returns:
            MulticanonicalResults: The results of the simulation.
        """
        start_multicanonical = time.perf_counter()

        # Get energy range
        normalized_energy_range_list = AlgorithmUtil.generate_energy_range_list(
            normalized_energy_range=initial_data.model.normalized_energy_range,
            num_divided_energy_range=parameters.num_divided_energy_range,
            overlap_rate=parameters.overlap_rate,
        )

        # Run the simulation
        print("Running multicanonical simulation by python ...", flush=True)
        start_simulation = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            result_list = list(
                executor.map(
                    _base_multicanonical,
                    repeat(initial_data.model),
                    repeat(parameters.num_sweeps),
                    repeat(
                        dict(
                            zip(
                                initial_data.normalized_energies, initial_data.entropies
                            )
                        )
                    ),
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
            modified_entropies,
            normalized_energies,
            histograms,
            order_parameters,
            energy_coeff,
        ) = _post_process(initial_data, result_list)

        result = MulticanonicalResults(
            initial_data=initial_data,
            parameters=parameters,
            entropies=modified_entropies,
            energies=normalized_energies * energy_coeff,
            normalized_energies=normalized_energies,
            histogram=histograms,
            order_parameters=order_parameters.to_order_parameter_results(energy_coeff),
            model=initial_data.model,
            info={
                "simulation_time": end_simulation - start_simulation,
                "total_time": time.perf_counter() - start_multicanonical,
                "backend": "py",
                "num_threads": num_threads,
                "calculate_order_parameters": calculate_order_parameters,
                "order_parameters": order_parameters,
            },
        )

        return result

    @staticmethod
    def multicanonical_cpp(
        initial_data: WangLandauResults,
        parameters: MulticanonicalParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
    ) -> MulticanonicalResults:
        """Run a multicanonical simulation by c++ implementation.

        Args:
            initial_data (WangLandauResults): The results of the primary simulation.
            parameters (MulticanonicalParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.

        Returns:
            MulticanonicalResults: The results of the simulation.
        """
        start_multicanonical = time.perf_counter()

        # Run the simulation
        print("Running multicanonical simulation by cpp ...", flush=True)
        start_simulation = time.perf_counter()
        result_list = cpp_muca.cpp_algorithm.run_multicanonical(
            model=cast_from_py_model(initial_data.model),
            parameters=cast_from_py_multicanonical_parameters(parameters),
            entropy_dict=dict(
                zip(initial_data.normalized_energies, initial_data.entropies)
            ),
            num_threads=num_threads,
            calculate_order_parameters=calculate_order_parameters,
        )
        end_simulation = time.perf_counter()
        print(
            f"Done simulation ({round(end_simulation - start_simulation, 1)}) [sec] by cpp",
            flush=True,
        )

        (
            modified_entropies,
            normalized_energies,
            histograms,
            order_parameters,
            energy_coeff,
        ) = _post_process(initial_data, result_list)

        result = MulticanonicalResults(
            initial_data=initial_data,
            parameters=parameters,
            entropies=modified_entropies,
            energies=normalized_energies * energy_coeff,
            normalized_energies=normalized_energies,
            histogram=histograms,
            order_parameters=order_parameters.to_order_parameter_results(energy_coeff),
            model=initial_data.model,
            info={
                "simulation_time": end_simulation - start_simulation,
                "total_time": time.perf_counter() - start_multicanonical,
                "backend": "cpp",
                "num_threads": num_threads,
                "calculate_order_parameters": calculate_order_parameters,
                "order_parameters": order_parameters,
            },
        )

        return result

    @classmethod
    def multicanonical(
        cls,
        initial_data: WangLandauResults,
        parameters: MulticanonicalParameters,
        num_threads: int = 1,
        calculate_order_parameters: bool = True,
        backend: str = "py",
    ) -> MulticanonicalResults:
        """Run a multicanonical simulation

        Args:
            initial_data (WangLandauResults): The results of the primary simulation.
            parameters (MulticanonicalParameters): The parameters for the simulation.
            num_threads (int, optional): The number of threads to use. Defaults to 1.
            calculate_order_parameters (bool, optional): Whether to calculate the order parameters. Defaults to True.
            backend (str, optional): The backend to use. "py" or "cpp". Defaults to "py".

        Raises:
            ValueError: If the backend is not supported.

        Returns:
            MulticanonicalResults: The results of the simulation.
        """
        if backend == "py":
            return cls.multicanonical_py(
                initial_data=initial_data,
                parameters=parameters,
                num_threads=num_threads,
                calculate_order_parameters=calculate_order_parameters,
            )
        elif backend == "cpp":
            return cls.multicanonical_cpp(
                initial_data=initial_data,
                parameters=parameters,
                num_threads=num_threads,
                calculate_order_parameters=calculate_order_parameters,
            )
        else:
            raise ValueError(f"Backend '{backend}' is not supported.")
