import random
from dataclasses import dataclass, field
from enum import Enum


class UpdateMethod(Enum):
    METROPOLIS = "METROPOLIS"
    HEAT_BATH = "HEAT_BATH"

    def __str__(self):
        return self.value

@dataclass
class WangLandauParameters:
    """Class to store the parameters of the Wang-Landau simulation.

    Attributes:
        modification_criterion (float): The modification criterion of the entropy difference.
        convergence_check_interval (int): The interval to check for convergence.
        flatness_criterion (float): The flatness criterion of the energy histogram.
        seed (int): The seed for the random number generator.
        reduce_rate (float): The rate to reduce the modification factor.
        max_sweeps (int): The maximum number of sweeps for the simulation.
        num_divided_energy_range (int): The number of divided energy ranges.
        overlap_rate (float): The overlap rate for the energy ranges.

    Raises:
        TypeError: If the types of the parameters are not valid.
        ValueError: If the values of the parameters are not valid.
    """

    modification_criterion: float = 1e-08
    convergence_check_interval: int = 1000
    flatness_criterion: float = 0.8
    seed: int = field(default_factory=lambda: random.randrange(2147483647))
    reduce_rate: float = 0.5
    max_sweeps: int = 9223372036854775807  # int64_t max
    num_divided_energy_range: int = 1
    overlap_rate: float = 0.2
    update_method: UpdateMethod = UpdateMethod.METROPOLIS

    def __post_init__(self):
        # Check if the types are valid
        if not isinstance(self.modification_criterion, float):
            raise TypeError("modification_criterion must be a float")
        if not isinstance(self.convergence_check_interval, int):
            raise TypeError("convergence_check_interval must be an int")
        if not isinstance(self.flatness_criterion, float):
            raise TypeError("flatness_criterion must be a float")
        if not isinstance(self.reduce_rate, float):
            raise TypeError("reduce_rate must be a float")
        if not isinstance(self.max_sweeps, int):
            raise TypeError("max_sweeps must be an int")
        if not isinstance(self.num_divided_energy_range, int):
            raise TypeError("num_energy_range must be an int")
        if not isinstance(self.overlap_rate, float):
            raise TypeError("overlap_rate must be a float")

        # Check if the values are valid
        if self.modification_criterion <= 0 or self.modification_criterion > 1:
            raise ValueError("modification_criterion must be in the range (0, 1]")
        if self.convergence_check_interval <= 0:
            raise ValueError("convergence_check_interval must be positive")
        if self.flatness_criterion < 0 or self.flatness_criterion >= 1:
            raise ValueError("flatness_criterion must be in the range [0, 1)")
        if self.reduce_rate <= 0 or self.reduce_rate >= 1:
            raise ValueError("reduce_rate must be in the range (0, 1)")
        if self.max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        if self.num_divided_energy_range <= 0:
            raise ValueError("num_divided_energy_range must be positive")
        if self.overlap_rate < 0 or self.overlap_rate > 1:
            raise ValueError("overlap_rate must be in the range [0, 1]")


@dataclass
class MulticanonicalParameters:
    """Class to store the parameters of the Multicanonical simulation.

    Attributes:
        num_sweeps (int): The number of sweeps for the simulation.
        num_divided_energy_range (int): The number of divided energy ranges.
        overlap_rate (float): The overlap rate for the energy ranges.
        seed (int): The seed for the random number generator.

    Raises:
        TypeError: If the types of the parameters are not valid.
        ValueError: If the values of the parameters are not valid.
    """

    num_sweeps: int = 1000
    num_divided_energy_range: int = 1
    overlap_rate: float = 0.2
    seed: int = field(default_factory=lambda: random.randrange(2147483647))

    def __post_init__(self):
        if not isinstance(self.num_sweeps, int):
            raise TypeError("num_sweeps must be an int")
        if not isinstance(self.num_divided_energy_range, int):
            raise TypeError("num_divided_energy_range must be an int")
        if not isinstance(self.overlap_rate, float):
            raise TypeError("overlap_rate must be a float")

        if self.num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")
        if self.num_divided_energy_range <= 0:
            raise ValueError("num_divided_energy_range must be positive")
        if self.overlap_rate < 0 or self.overlap_rate > 1:
            raise ValueError("overlap_rate must be in the range [0, 1]")
