from typing import Union

import mpmath
import numpy as np

from muca.results.simulation_result import MulticanonicalResults, WangLandauResults


class Analyzer:

    @classmethod
    def calculate_internal_energy(
        cls,
        data: Union[WangLandauResults, MulticanonicalResults],
        temperature_list: np.ndarray,
        power: int = 1,
    ) -> np.ndarray:
        """Calculate the internal energy of the system.

        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (np.ndarray): The list of temperatures to calculate the internal energy for.
            power (int, optional): The power to raise the energy to. Defaults to 1.

        Returns:
            np.ndarray: The internal energy of the system for given temperatures.
        """
        internal_energies = np.zeros(len(temperature_list))
        for i, temperature in enumerate(temperature_list):
            exponents = -data.energies / temperature + data.entropies
            exponents -= np.max(exponents)
            partition_function = mpmath.mpf(0)
            expectation = mpmath.mpf(0)
            for e, s in zip(data.energies, exponents):
                partition_function += mpmath.exp(s)
                expectation += (e**power) * mpmath.exp(s)
            internal_energies[i] = float(expectation / partition_function)
        return internal_energies

    @classmethod
    def calculate_heat_capacity(
        cls,
        data: Union[WangLandauResults, MulticanonicalResults],
        temperature_list: np.ndarray,
    ) -> np.ndarray:
        """Calculate the heat capacity of the system.
        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (np.ndarray): The list of temperatures to calculate the heat capacity for.
        Returns:
            np.ndarray: The heat capacity of the system for given temperatures.
        """
        e = cls.calculate_internal_energy(data, temperature_list)
        ee = cls.calculate_internal_energy(data, temperature_list, power=2)
        return (ee - e**2) / (temperature_list**2)

    @classmethod
    def calculate_ordered_Q_squared_fourier_intensity(
        cls,
        data: Union[WangLandauResults, MulticanonicalResults],
        temperature_list: np.ndarray,
    ) -> np.ndarray:
        """Calculate the order parameter from its distribution obtained from the simulation.
        This is defined by the total of specific squared components from the absolute value of the Fourier transform of the spin configuration.

        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (Union[np.ndarray, list[float]]): The list of temperatures to calculate the order parameter for.
        Returns:
            np.ndarray: The order parameter of the system.
        """
        intensity = np.zeros(len(temperature_list))
        for i, temperature in enumerate(temperature_list):
            exponents = -data.energies / temperature + data.entropies
            exponents -= np.max(exponents)
            partition_function = mpmath.mpf(0)
            expectation = mpmath.mpf(0)
            for j, s in enumerate(exponents):
                partition_function += mpmath.exp(s)
                expectation += data.order_parameters.abs_fourier_second[j] * mpmath.exp(
                    s
                )
            intensity[i] = float(expectation / partition_function)
        return intensity

    @classmethod
    def calculate_ordered_Q_quartic_fourier_intensity(
        cls,
        data: Union[WangLandauResults, MulticanonicalResults],
        temperature_list: np.ndarray,
    ) -> np.ndarray:
        """Calculate the squared order parameter from its distribution obtained from the simulation.
        Here the order parameter is defined by the total of specific squared components
        from the absolute value of the Fourier transform of the spin configuration.

        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (Union[np.ndarray, list[float]]): The list of temperatures to calculate the order parameter for.

        Returns:
            np.ndarray: The order parameter of the system.
        """
        intensity = np.zeros(len(temperature_list))
        for i, temperature in enumerate(temperature_list):
            exponents = -data.energies / temperature + data.entropies
            exponents -= np.max(exponents)
            partition_function = mpmath.mpf(0)
            expectation = mpmath.mpf(0)
            for j, s in enumerate(exponents):
                partition_function += mpmath.exp(s)
                expectation += data.order_parameters.abs_fourier_fourth[j] * mpmath.exp(
                    s
                )
            intensity[i] = float(expectation / partition_function)
        return intensity

    @classmethod
    def calculate_binder_cumulant(
        cls,
        data: Union[WangLandauResults, MulticanonicalResults],
        temperature_list: np.ndarray,
    ) -> np.ndarray:
        """Calculate the Binder cumulant of the order parameter.
        Here the order parameter is defined by the total of specific squared components
        from the absolute value of the Fourier transform of the spin configuration.

        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (Union[np.ndarray, list[float]]): The list of temperatures to calculate the order parameter for.

        Returns:
            np.ndarray: The Binder cumulant of the order parameter.
        """
        mm = cls.calculate_ordered_Q_squared_fourier_intensity(data, temperature_list)
        mmmm = cls.calculate_ordered_Q_quartic_fourier_intensity(data, temperature_list)
        return 1 - mmmm / (3 * mm**2)
