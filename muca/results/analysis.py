import mpmath
import numpy as np


class Analyzer:

    @classmethod
    def calculate_expectation(
        cls,
        energies: np.ndarray,
        entropies: np.ndarray,
        temperature_list: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Calculate the expectation value of a given observable.

        Args:
            energies (np.ndarray): The energies of the system.
            entropies (np.ndarray): The entropies of the system.
            temperature_list (np.ndarray): The list of temperatures to calculate the expectation value for.
            values (np.ndarray): The values of the observable.

        Returns:
            np.ndarray: The expectation value of the observable for given temperatures.
        """
        result = np.zeros(len(temperature_list))
        for i, temperature in enumerate(temperature_list):
            exponents = -energies / temperature + entropies
            exponents -= np.max(exponents)
            partition_function = mpmath.mpf(0)
            expectation = mpmath.mpf(0)
            for j, s in enumerate(exponents):
                partition_function += mpmath.exp(s)
                expectation += values[j] * mpmath.exp(s)
            result[i] = float(expectation / partition_function)
        return result

    @classmethod
    def calculate_energy_expectation(
        cls,
        energies: np.ndarray,
        entropies: np.ndarray,
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
            exponents = -energies / temperature + entropies
            exponents -= np.max(exponents)
            partition_function = mpmath.mpf(0)
            expectation = mpmath.mpf(0)
            for e, s in zip(energies, exponents):
                partition_function += mpmath.exp(s)
                expectation += (e**power) * mpmath.exp(s)
            internal_energies[i] = float(expectation / partition_function)
        return internal_energies

    @classmethod
    def calculate_heat_capacity(
        cls,
        energies: np.ndarray,
        entropies: np.ndarray,
        temperature_list: np.ndarray,
    ) -> np.ndarray:
        """Calculate the heat capacity of the system.
        Args:
            data (Union[WangLandauResults, MulticanonicalResults]): The data obtained from the simulation.
            temperature_list (np.ndarray): The list of temperatures to calculate the heat capacity for.
        Returns:
            np.ndarray: The heat capacity of the system for given temperatures.
        """
        e = cls.calculate_energy_expectation(energies, entropies, temperature_list)
        ee = cls.calculate_energy_expectation(
            energies, entropies, temperature_list, power=2
        )
        return (ee - e**2) / (temperature_list**2)
