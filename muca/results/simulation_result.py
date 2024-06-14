import pickle
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import numpy as np

from muca.algorithm.parameters import MulticanonicalParameters, WangLandauParameters
from muca.model.p_body_ising import PBodyTwoDimIsing


@dataclass
class OrderParameterResults:
    """Class to store the order parameters of the system.

    Attributes:
        squared_magnetization (Optional[np.ndarray]): The squared magnetization of the system.
        forth_magnetization (Optional[np.ndarray]): The forth magnetization of the system.
        abs_fourier_second (Optional[np.ndarray]): The absolute value of the second Fourier transform of the order parameter.
        abs_fourier_fourth (Optional[np.ndarray]): The absolute value of the fourth Fourier transform of the order parameter.
        energies (Optional[np.ndarray]): The energies of the system.
        normalized_energies (Optional[np.ndarray]): The normalized energies of the system.
            The normalized energies are the energies that the values of the spins are twice the value of the spin.
    """

    squared_magnetization: Optional[np.ndarray] = None
    forth_magnetization: Optional[np.ndarray] = None
    abs_fourier_second: Optional[np.ndarray] = None
    abs_fourier_fourth: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    normalized_energies: Optional[np.ndarray] = None

    def __post_init__(self):
        # Check if the results are not zero-size arrays
        if (
            self.squared_magnetization is not None
            and self.squared_magnetization.size == 0
        ):
            raise ValueError("The squared magnetization array is empty.")
        if self.forth_magnetization is not None and self.forth_magnetization.size == 0:
            raise ValueError("The forth magnetization array is empty.")
        if self.abs_fourier_second is not None and self.abs_fourier_second.size == 0:
            raise ValueError(
                "The absolute value of the second Fourier transform array is empty."
            )
        if self.abs_fourier_fourth is not None and self.abs_fourier_fourth.size == 0:
            raise ValueError(
                "The absolute value of the fourth Fourier transform array is empty."
            )
        if self.energies is not None and self.energies.size == 0:
            raise ValueError("The energies array is empty.")
        if self.normalized_energies is not None and self.normalized_energies.size == 0:
            raise ValueError("The normalized energies array is empty.")


@dataclass
class WangLandauResults:
    """Class to store the results of the Wang-Landau simulation.

    Attributes:
        parameters (Optional[WangLandauParameters]): The parameters of the simulation.
        entropies (Optional[np.ndarray]): The entropies of the system.
        energies (Optional[np.ndarray]): The energies of the system.
        normalized_energies (Optional[np.ndarray]): The normalized energies of the system.
        total_sweeps (Optional[int]): The total number of sweeps performed in the simulation.
        final_modification_factor (Optional[float]): The final modification factor of the simulation.
        order_parameters (Optional[OrderParameterResults]): The order parameters of the system.
        model (Optional[PBodyTwoDimIsing]): The model used in the simulation.
        info (Optional[dict]): Additional information of the simulation.
    """

    # Simulation Parameters
    parameters: Optional[WangLandauParameters] = None

    # Simulation Results
    entropies: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    normalized_energies: Optional[np.ndarray] = None
    total_sweeps: Optional[int] = None
    final_modification_factor: Optional[float] = None
    order_parameters: Optional[OrderParameterResults] = None

    # Model Information
    model: Optional[PBodyTwoDimIsing] = None

    # Additional Information
    info: Optional[dict] = None

    def store_as_pickle(self, file_path: str) -> None:
        """Store the results of the simulation as a pickle file.
        Args:
            file_path (str): The path to store the results.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(file_path: str) -> "WangLandauResults":
        """Load the results of the simulation from a pickle file.
        Args:
            file_path (str): The path to load the results from.
        Returns:
            WangLandauResults: The results of the simulation.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)


@dataclass
class MulticanonicalResults:
    """Class to store the results of the Multicanonical simulation.

    Attributes:
        initial_data (Optional[WangLandauResults]): The initial data used to start the simulation.
        parameters (Optional[MulticanonicalParameters]): The parameters of the simulation.
        entropies (Optional[np.ndarray]): The entropies of the system.
        energies (Optional[np.ndarray]): The energies of the system.
        normalized_energies (Optional[np.ndarray]): The normalized energies of the system.
        histogram (Optional[np.ndarray]): The histogram of the system.
        order_parameters (Optional[OrderParameterResults]): The order parameters of the system.
        model (Optional[PBodyTwoDimIsing]): The model used in the simulation.
        info (Optional[dict]): Additional information of the simulation.
    """

    # Simulation Parameters
    initial_data: Optional[WangLandauResults] = None
    parameters: Optional[MulticanonicalParameters] = None

    # Simulation Results
    entropies: Optional[np.ndarray] = None
    energies: Optional[np.ndarray] = None
    normalized_energies: Optional[np.ndarray] = None
    histogram: Optional[np.ndarray] = None
    order_parameters: Optional[OrderParameterResults] = None

    # Model Information
    model: Optional[PBodyTwoDimIsing] = None

    # Additional Information
    info: Optional[dict] = None

    def store_as_pickle(self, file_path: str) -> None:
        """Store the results of the simulation as a pickle file.
        Args:
            file_path (str): The path to store the results.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_pickle(file_path: str) -> "MulticanonicalResults":
        """Load the results of the simulation from a pickle file.
        Args:
            file_path (str): The path to load the results from.
        Returns:
            MulticanonicalResults: The results of the simulation.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
