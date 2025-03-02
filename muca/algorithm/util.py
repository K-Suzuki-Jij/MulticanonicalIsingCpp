import copy
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from typing import Union

import numpy as np

from muca.model.p_body_ising import PBodyTwoDimIsing
from muca.results.simulation_result import OrderParameterResults


@dataclass
class OrderParameters:
    """Class to store the order parameter distribution of the system.

    Attributes:
        mag_2 (dict[int, float]): The squared magnetization of the system.
        mag_4 (dict[int, float]): The fourth power of the magnetization of the system.
        abs_f2 (dict[int, float]): The absolute value of the second Fourier transform of the order parameter.
        abs_f4 (dict[int, float]): The absolute value of the fourth Fourier transform of the order parameter.
        normalized_energy_count (dict[int, int]): The number of times an energy was visited.
    """

    mag_2: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    mag_4: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    abs_f2: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    abs_f4: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    normalized_energy_count: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def clear(self) -> None:
        """Clear the order parameters."""
        self.mag_2.clear()
        self.mag_4.clear()
        self.abs_f2.clear()
        self.abs_f4.clear()
        self.normalized_energy_count.clear()

    def copy(self) -> "OrderParameters":
        """Copy the OrderParameters.

        Returns:
            OrderParameters: The copied OrderParameters.
        """
        return OrderParameters(
            mag_2=copy.deepcopy(self.mag_2),
            mag_4=copy.deepcopy(self.mag_4),
            abs_f2=copy.deepcopy(self.abs_f2),
            abs_f4=copy.deepcopy(self.abs_f4),
            normalized_energy_count=copy.deepcopy(self.normalized_energy_count),
        )

    def assign(self, normalized_energy: int, other: "OrderParameters") -> None:
        """Assign the values of the other OrderParameters to this OrderParameters.

        Args:
            normalized_energy (int): The normalized energy of the system.
            other (OrderParameters): The other OrderParameters.
        """
        self.mag_2[normalized_energy] = other.mag_2[normalized_energy]
        self.mag_4[normalized_energy] = other.mag_4[normalized_energy]
        self.abs_f2[normalized_energy] = other.abs_f2[normalized_energy]
        self.abs_f4[normalized_energy] = other.abs_f4[normalized_energy]
        self.normalized_energy_count[normalized_energy] = other.normalized_energy_count[
            normalized_energy
        ]

    def to_order_parameter_results(
        self, energy_coefficient: float
    ) -> OrderParameterResults:
        """Convert the OrderParameters to OrderParameterResults.

        Args:
            energy_coefficient (float): The coefficient to calculate energies.
            ordered_Q (np.ndarray): The ordered Fourier components of the system.

        Returns:
            OrderParameterResults: The order parameters of the system.
        """
        if len(self.normalized_energy_count) == 0:
            return OrderParameterResults()

        energies = (
            np.array(list(self.normalized_energy_count.keys())) * energy_coefficient
        )

        _, sorted_mag_2 = zip(*sorted(zip(energies, self.mag_2.values())))
        _, sorted_mag_4 = zip(*sorted(zip(energies, self.mag_4.values())))
        _, sorted_abs_f2 = zip(*sorted(zip(energies, self.abs_f2.values())))
        _, sorted_abs_f4 = zip(*sorted(zip(energies, self.abs_f4.values())))
        normalized_energies, counts = zip(
            *sorted(zip(energies, self.normalized_energy_count.values()))
        )

        return OrderParameterResults(
            squared_magnetization=np.array(sorted_mag_2) / np.array(counts),
            forth_magnetization=np.array(sorted_mag_4) / np.array(counts),
            abs_fourier_second=np.array(sorted_abs_f2) / np.array(counts),
            abs_fourier_fourth=np.array(sorted_abs_f4) / np.array(counts),
            energies=np.array(normalized_energies) * energy_coefficient,
            normalized_energies=np.array(normalized_energies),
        )


class OrderParameterCounter:
    """Class to count the order parameters when multicanonical simulation is performed.

    Attributes:
        model (PBodyTwoDimIsing): The model of the system.
        op_coeff (float): The coefficient of the order parameter.
        fourier (np.ndarray): The Fourier transform of the order parameter.
        order_parameters (OrderParameters): The order parameters of the system.
        ordered_Q (np.ndarray): The ordered Fourier components of the system.
    """

    def __init__(self, model: PBodyTwoDimIsing, initial_spins: np.ndarray) -> None:
        """Initialize the OrderParameterCounter.

        Args:
            model (PBodyTwoDimIsing): The model of the system.
            initial_spins (np.ndarray): The initial state of the system.
        """
        self.model: PBodyTwoDimIsing = model
        self.op_coeff: float = model.spin_scale_factor / 2 / (model.Lx * model.Ly)
        self.fourier = np.fft.fft2(initial_spins, norm="backward") * self.op_coeff
        self.order_parameters = OrderParameters()
        self.ordered_Q = np.array(
            np.meshgrid(
                np.arange(model.p) * model.Lx // model.p,
                np.arange(model.p) * model.Ly // model.p,
            )
        ).T.reshape(-1, 2)

    def update_fourier(self, dS: float, x: int, y: int) -> None:
        """Update the Fourier transform of the order parameter.

        Args:
            dS (float): The change in the entropy.
            x (int): The x-coordinate of the site.
            y (int): The y-coordinate of the site.
        """
        exponential_terms = np.exp(
            -2
            * np.pi
            * 1j
            * (
                self.ordered_Q[:, 0] * x / self.model.Lx
                + self.ordered_Q[:, 1] * y / self.model.Ly
            )
        )
        self.fourier[self.ordered_Q[:, 0], self.ordered_Q[:, 1]] += (
            self.op_coeff * dS * exponential_terms
        )

    def update_order_parameters(self, normalized_energy: int) -> None:
        """Update the order parameters of the system.

        Args:
            normalized_energy (int): The normalized energy of the system.
        """
        temp_abs_f2 = np.sum(
            np.abs(self.fourier[self.ordered_Q[:, 0], self.ordered_Q[:, 1]]) ** 2
        )
        sq_mag = np.abs(self.fourier[0, 0]) ** 2
        self.order_parameters.abs_f2[normalized_energy] += temp_abs_f2
        self.order_parameters.abs_f4[normalized_energy] += temp_abs_f2**2
        self.order_parameters.mag_2[normalized_energy] += sq_mag
        self.order_parameters.mag_4[normalized_energy] += sq_mag**2
        self.order_parameters.normalized_energy_count[normalized_energy] += 1

    def to_order_parameters(self) -> OrderParameters:
        """Convert the OrderParameterCounter to OrderParameters.

        Returns:
            OrderParameters: The order parameters of the system.
        """
        return self.order_parameters


class AlgorithmUtil:
    """Utility class for the multicanonical and Wang-Landau algorithms."""

    @staticmethod
    def generate_initial_state(
        model: PBodyTwoDimIsing, normalized_energy_range: tuple[int, int]
    ) -> np.ndarray:
        """Generate an initial state for the system by wang-landau method.
        Each state is stored as twice the value of the spin.

        Args:
            model (PBodyTwoDimIsing): The model of the system.
            normalized_energy_range (tuple[int, int]): The range of the energy. Defaults to None.
                Here, the energy is assumed that the value of the spin takes twice the value of the spin.

        Raises:
            ValueError: If the state with the desired energy was not found.

        Returns:
            np.ndarray: The initial state of the system.
        """
        e_min, e_max = normalized_energy_range
        entropy_dict: dict[int, float] = defaultdict(float)
        twice_spins = np.random.choice(model.twice_spin_set, size=(model.Lx, model.Ly))
        dE = model.make_energy_difference(twice_spins)
        normalized_energy = model.calculate_normalized_energy(twice_spins)
        for _ in range(1000000):
            for x, y in product(range(model.Lx), range(model.Ly)):
                new_spin_value = np.random.choice(
                    model.twice_spin_set[model.twice_spin_set != twice_spins[x, y]]
                )
                normalized_new_energy = normalized_energy + dE[x, y] * (
                    new_spin_value - twice_spins[x, y]
                )

                # If the new state has the desired energy, return it
                if e_min <= normalized_new_energy <= e_max:
                    twice_spins[x, y] = new_spin_value
                    return twice_spins

                # Do the Wang-Landau update
                dS = (
                    entropy_dict[normalized_new_energy]
                    - entropy_dict[normalized_energy]
                )
                if dS <= 0 or np.random.rand() < np.exp(-dS):
                    model.update_energy_difference(
                        new_spin_value, x, y, dE, twice_spins
                    )
                    twice_spins[x, y] = new_spin_value
                    normalized_energy = normalized_new_energy
                entropy_dict[normalized_energy] += 1.0

        raise ValueError("The state with the desired energy was not found.")

    @staticmethod
    def generate_energy_range_list(
        normalized_energy_range: tuple[int, int],
        num_divided_energy_range: int,
        overlap_rate: float,
    ) -> list[tuple[float, float]]:
        """Generate a list of divided energy ranges.

        Args:
            normalized_energy_range (tuple[int, int]): The normalized energy range.
            num_divided_energy_range (int): The number of divided energy ranges.
            overlap_rate (float): The overlap rate of the divided energy ranges.

        Returns:
            list[tuple[float, float]]: The list of divided energy ranges.
        """
        e_min, e_max = normalized_energy_range
        length = (e_max - e_min) / (
            num_divided_energy_range - overlap_rate * (num_divided_energy_range - 1)
        )
        divided_energy_range_list = []
        for i in range(num_divided_energy_range):
            e_left = e_min + i * length * (1 - overlap_rate)
            e_right = e_left + length
            divided_energy_range_list.append([e_left, e_right])

        # 数値誤差を考えて区間の端は改めて最小/最大エネルギーを登録する
        divided_energy_range_list[0][0] = e_min
        divided_energy_range_list[-1][1] = e_max
        return divided_energy_range_list

    @staticmethod
    def merge_results(
        normalized_energy_range: tuple[int, int],
        separated_data_list: list[dict[int, float]],
        order_parameters_list: list[OrderParameters] = None,
        connect_edge: bool = False,
    ) -> tuple[dict[int, Union[int, float]], OrderParameters]:
        """Merge the results of the multicanonical or Wang-Landau simulation.

        Args:
            normalized_energy_range (tuple[int, int]): The normalized energy range.
            separated_data_list (list[dict[int, float]]): The list of separated data.
            order_parameters_list (list[OrderParameters], optional): The list of order parameters. Defaults to None.
            connect_edge (bool, optional): Whether to connect the edges of the energy ranges. Defaults to False.

        Returns:
            tuple[dict[int, Union[int, float]], OrderParameters]: The merged data and order parameters.
        """
        e_min, e_max = normalized_energy_range
        num_energy_range: int = len(separated_data_list)
        merged_data: dict[int, float] = {}
        merged_order_parameters = OrderParameters()
        bias: float = 0.0

        for i in range(num_energy_range):
            e_l = e_min + i * (e_max - e_min) / num_energy_range
            e_r = e_min + (i + 1) * (e_max - e_min) / num_energy_range
            for normalized_energy, entropy in separated_data_list[i].items():
                if e_l <= normalized_energy <= e_r:
                    merged_data[normalized_energy] = entropy - bias
                    if len(order_parameters_list[i].normalized_energy_count) > 0:
                        merged_order_parameters.assign(
                            normalized_energy, order_parameters_list[i]
                        )
            if connect_edge and i < num_energy_range - 1:
                energies = np.array(list(separated_data_list[i].keys()))
                next_energies = np.array(list(separated_data_list[i + 1].keys()))
                next_data = np.array(list(separated_data_list[i + 1].values()))
                target_energy = min(
                    energies[energies <= e_r],
                    key=lambda x: abs(x - e_r),
                    default=None,
                )
                bias = (
                    next_data[next_energies == target_energy][0]
                    - merged_data[target_energy]
                )

        return merged_data, merged_order_parameters

    @staticmethod
    def scale_entropies(entropies: list, model: PBodyTwoDimIsing) -> np.ndarray:
        entropies = np.array(entropies) - np.max(entropies)
        bias = reduce(np.logaddexp, entropies) - model.Lx * model.Ly * np.log(
            int(2 * model.spin + 1)
        )
        entropies -= bias
        return entropies
