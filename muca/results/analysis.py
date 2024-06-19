from functools import reduce

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
        min_value = np.min(values) - 0.1
        scaled_values = values - min_value
        result = np.zeros(len(temperature_list))
        for i, temperature in enumerate(temperature_list):
            exponents = -energies / temperature + entropies
            exponents -= np.max(exponents)
            a = reduce(np.logaddexp, exponents + np.log(scaled_values))
            b = reduce(np.logaddexp, exponents)
            result[i] = np.exp(a - b) + min_value
        return result

    @classmethod
    def calculate_expectation_mpmath(
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
        e = cls.calculate_expectation(energies, entropies, temperature_list, energies)
        ee = cls.calculate_expectation(
            energies, entropies, temperature_list, energies**2
        )
        return (ee - e**2) / (temperature_list**2)
