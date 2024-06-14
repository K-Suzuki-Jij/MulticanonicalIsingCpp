import itertools as it

import numpy as np


class PBodyTwoDimIsing:
    """
    Represents a two-dimensional p-body Ising model with specific physical properties.

    Attributes:
        J (float): Value of the interaction.
        p (int): The number of spins in the interaction. Must be larger than 1.
        Lx (int): The size of the lattice in the x-direction. Must be non-negative.
        Ly (int): The size of the lattice in the y-direction. Must be non-negative.
        spin (float): Magnitude of the spin. Must be a non-negative half-integer.
        spin_scale_factor (float): Scale factor for the spin. Must be positive.
        normalized_energy_range (tuple[int, int]): The range of normalized energies.
        twice_spin_set (np.ndarray): The set of twice the spin values.

    Raises:
        ValueError: If any of the conditions for the parameters are not met.
    """

    def __init__(
        self, J: float, p: int, Lx: int, Ly: int, spin: float, spin_scale_factor: float
    ) -> None:
        # Check if the types are valid
        if not isinstance(p, int):
            raise TypeError("p must be an int")
        if not isinstance(Lx, int):
            raise TypeError("Lx must be an int")
        if not isinstance(Ly, int):
            raise TypeError("Ly must be an int")

        # Check if the values are valid
        if J == 0:
            raise ValueError("J must be non-zero")
        if p < 2:
            raise ValueError("p must be larger than 1")
        if Lx < 0:
            raise ValueError("Lx must be non-negative")
        if Ly < 0:
            raise ValueError("Ly must be non-negative")
        if spin % 0.5 != 0 or spin < 0.5:
            raise ValueError("Spin must be a non-negative half-integer")
        if spin_scale_factor <= 0:
            raise ValueError("Spin scale factor must be positive")

        # Set values
        self.J: float = J
        self.p: int = p
        self.Lx: int = Lx
        self.Ly: int = Ly
        self.spin: float = spin
        self.spin_scale_factor: float = spin_scale_factor
        self.normalized_energy_range: tuple[int, int] = (
            -2 * self.Lx * self.Ly * (int(2 * self.spin) ** self.p),
            +2 * self.Lx * self.Ly * (int(2 * self.spin) ** self.p),
        )
        self.twice_spin_set = np.arange(-int(2 * self.spin), int(2 * self.spin) + 1, 2)

    def calculate_normalized_energy(self, spin_configuration: np.ndarray) -> int:
        """Calculate normalized energy of the spin configuration.
        Note that J and spin_scale_factor are assumed to be 1.

        Args:
            spin_configuration (np.ndarray): The spin configuration as a two dimensional array.

        Returns:
            float: The energy of the spin configuration.
        """
        energy = 0
        offset = np.arange(self.p)
        for x, y in it.product(range(self.Lx), range(self.Ly)):
            spin_prod_x = np.prod(spin_configuration[(x + offset) % self.Lx, y])
            spin_prod_y = np.prod(spin_configuration[x, (y + offset) % self.Ly])
            energy += spin_prod_x + spin_prod_y

        return energy * (int((self.J > 0) - (self.J < 0)))

    def make_energy_difference(self, spin_configuration: np.ndarray) -> np.ndarray:
        """Make the energy difference for the spin configuration.
        Note that the value of interaction J and the difference of spin value is assumed to be 1.

        Args:
            spin_configuration (np.ndarray): The spin configuration as a two dimensional array.

        Returns:
            np.ndarray: The energy difference as a two dimensional array.
        """
        sign = int((self.J > 0) - (self.J < 0))
        energy_difference = np.zeros((self.Lx, self.Ly), dtype=np.int64)
        for x, y in it.product(range(self.Lx), range(self.Ly)):
            x_indices = [
                (x + dx) % self.Lx for dx in range(-self.p + 1, self.p) if dx != 0
            ]
            y_indices = [
                (y + dy) % self.Ly for dy in range(-self.p + 1, self.p) if dy != 0
            ]
            sequences_x = [x_indices[j : j + (self.p - 1)] for j in range(self.p)]
            sequences_y = [y_indices[j : j + (self.p - 1)] for j in range(self.p)]

            for i in range(self.p):
                spin_prod_x = np.prod(spin_configuration[sequences_x[i], y])
                spin_prod_y = np.prod(spin_configuration[x, sequences_y[i]])
                energy_difference[x, y] += sign * (spin_prod_x + spin_prod_y)

        return energy_difference

    def update_energy_difference(
        self,
        new_spin_value: float,
        x: int,
        y: int,
        energy_difference: np.ndarray,
        spin_configuration: np.ndarray,
    ) -> None:
        """Update the energy difference according to the new spin value at the position (x, y).

        Args:
            new_spin_value (int): The new spin value.
            x (int): The x position.
            y (int): The y position.
            energy_difference (np.ndarray): The energy difference as a two dimensional array.
            spin_configuration (np.ndarray): The spin configuration as a two dimensional array.

        Raises:
            ValueError: If the value of p is not 2, 3, 4, or 5.
        """
        if new_spin_value == spin_configuration[x, y]:
            return

        # Rename some variables for clarity
        dS = (new_spin_value - spin_configuration[x, y]) * int(
            (self.J > 0) - (self.J < 0)
        )
        dE = energy_difference
        S = spin_configuration
        Lx = self.Lx
        Ly = self.Ly

        if self.p == 2:
            dE[(x - 1) % Lx, y] += dS
            dE[(x + 1) % Lx, y] += dS
            dE[x, (y - 1) % Ly] += dS
            dE[x, (y + 1) % Ly] += dS
        elif self.p == 3:
            dE[(x - 2) % Lx, y] += dS * S[(x - 1) % Lx, y]
            dE[(x - 1) % Lx, y] += dS * (S[(x - 2) % Lx, y] + S[(x + 1) % Lx, y])
            dE[(x + 1) % Lx, y] += dS * (S[(x - 1) % Lx, y] + S[(x + 2) % Lx, y])
            dE[(x + 2) % Lx, y] += dS * S[(x + 1) % Lx, y]
            dE[x, (y - 2) % Ly] += dS * S[x, (y - 1) % Ly]
            dE[x, (y - 1) % Ly] += dS * (S[x, (y - 2) % Ly] + S[x, (y + 1) % Ly])
            dE[x, (y + 1) % Ly] += dS * (S[x, (y - 1) % Ly] + S[x, (y + 2) % Ly])
            dE[x, (y + 2) % Ly] += dS * S[x, (y + 1) % Ly]
        elif self.p == 4:
            x_m3, x_m2, x_m1 = ((x - 3) % Lx, (x - 2) % Lx, (x - 1) % Lx)
            x_p1, x_p2, x_p3 = ((x + 1) % Lx, (x + 2) % Lx, (x + 3) % Lx)
            y_m3, y_m2, y_m1 = ((y - 3) % Ly, (y - 2) % Ly, (y - 1) % Ly)
            y_p1, y_p2, y_p3 = ((y + 1) % Ly, (y + 2) % Ly, (y + 3) % Ly)
            dE[x_m3, y] += dS * S[x_m2, y] * S[x_m1, y]
            dE[x_m2, y] += dS * (S[x_m3, y] * S[x_m1, y] + S[x_m1, y] * S[x_p1, y])
            dE[x_m1, y] += dS * (
                S[x_m3, y] * S[x_m2, y]
                + S[x_m2, y] * S[x_p1, y]
                + S[x_p1, y] * S[x_p2, y]
            )
            dE[x_p1, y] += dS * (
                S[x_m2, y] * S[x_m1, y]
                + S[x_m1, y] * S[x_p2, y]
                + S[x_p2, y] * S[x_p3, y]
            )
            dE[x_p2, y] += dS * (S[x_m1, y] * S[x_p1, y] + S[x_p1, y] * S[x_p3, y])
            dE[x_p3, y] += dS * S[x_p1, y] * S[x_p2, y]
            dE[x, y_m3] += dS * S[x, y_m2] * S[x, y_m1]
            dE[x, y_m2] += dS * (S[x, y_m3] * S[x, y_m1] + S[x, y_m1] * S[x, y_p1])
            dE[x, y_m1] += dS * (
                S[x, y_m3] * S[x, y_m2]
                + S[x, y_m2] * S[x, y_p1]
                + S[x, y_p1] * S[x, y_p2]
            )
            dE[x, y_p1] += dS * (
                S[x, y_m2] * S[x, y_m1]
                + S[x, y_m1] * S[x, y_p2]
                + S[x, y_p2] * S[x, y_p3]
            )
            dE[x, y_p2] += dS * (S[x, y_m1] * S[x, y_p1] + S[x, y_p1] * S[x, y_p3])
            dE[x, y_p3] += dS * S[x, y_p1] * S[x, y_p2]
        elif self.p == 5:
            x_m4, x_m3, x_m2, x_m1 = (
                (x - 4) % Lx,
                (x - 3) % Lx,
                (x - 2) % Lx,
                (x - 1) % Lx,
            )
            x_p1, x_p2, x_p3, x_p4 = (
                (x + 1) % Lx,
                (x + 2) % Lx,
                (x + 3) % Lx,
                (x + 4) % Lx,
            )
            y_m4, y_m3, y_m2, y_m1 = (
                (y - 4) % Ly,
                (y - 3) % Ly,
                (y - 2) % Ly,
                (y - 1) % Ly,
            )
            y_p1, y_p2, y_p3, y_p4 = (
                (y + 1) % Ly,
                (y + 2) % Ly,
                (y + 3) % Ly,
                (y + 4) % Ly,
            )
            dE[x_m4, y] += dS * S[x_m3, y] * S[x_m2, y] * S[x_m1, y]
            dE[x_m3, y] += dS * (
                S[x_m4, y] * S[x_m2, y] * S[x_m1, y]
                + S[x_m2, y] * S[x_m1, y] * S[x_p1, y]
            )
            dE[x_m2, y] += dS * (
                S[x_m4, y] * S[x_m3, y] * S[x_m1, y]
                + S[x_m3, y] * S[x_m1, y] * S[x_p1, y]
                + S[x_m1, y] * S[x_p1, y] * S[x_p2, y]
            )
            dE[x_m1, y] += dS * (
                S[x_m4, y] * S[x_m3, y] * S[x_m2, y]
                + S[x_m3, y] * S[x_m2, y] * S[x_p1, y]
                + S[x_m2, y] * S[x_p1, y] * S[x_p2, y]
                + S[x_p1, y] * S[x_p2, y] * S[x_p3, y]
            )
            dE[x_p1, y] += dS * (
                S[x_m3, y] * S[x_m2, y] * S[x_m1, y]
                + S[x_m2, y] * S[x_m1, y] * S[x_p2, y]
                + S[x_m1, y] * S[x_p2, y] * S[x_p3, y]
                + S[x_p2, y] * S[x_p3, y] * S[x_p4, y]
            )
            dE[x_p2, y] += dS * (
                S[x_m2, y] * S[x_m1, y] * S[x_p1, y]
                + S[x_m1, y] * S[x_p1, y] * S[x_p3, y]
                + S[x_p1, y] * S[x_p3, y] * S[x_p4, y]
            )
            dE[x_p3, y] += dS * (
                S[x_m1, y] * S[x_p1, y] * S[x_p2, y]
                + S[x_p1, y] * S[x_p2, y] * S[x_p4, y]
            )
            dE[x_p4, y] += dS * S[x_p1, y] * S[x_p2, y] * S[x_p3, y]
            dE[x, y_m4] += dS * S[x, y_m3] * S[x, y_m2] * S[x, y_m1]
            dE[x, y_m3] += dS * (
                S[x, y_m4] * S[x, y_m2] * S[x, y_m1]
                + S[x, y_m2] * S[x, y_m1] * S[x, y_p1]
            )
            dE[x, y_m2] += dS * (
                S[x, y_m4] * S[x, y_m3] * S[x, y_m1]
                + S[x, y_m3] * S[x, y_m1] * S[x, y_p1]
                + S[x, y_m1] * S[x, y_p1] * S[x, y_p2]
            )
            dE[x, y_m1] += dS * (
                S[x, y_m4] * S[x, y_m3] * S[x, y_m2]
                + S[x, y_m3] * S[x, y_m2] * S[x, y_p1]
                + S[x, y_m2] * S[x, y_p1] * S[x, y_p2]
                + S[x, y_p1] * S[x, y_p2] * S[x, y_p3]
            )
            dE[x, y_p1] += dS * (
                S[x, y_m3] * S[x, y_m2] * S[x, y_m1]
                + S[x, y_m2] * S[x, y_m1] * S[x, y_p2]
                + S[x, y_m1] * S[x, y_p2] * S[x, y_p3]
                + S[x, y_p2] * S[x, y_p3] * S[x, y_p4]
            )
            dE[x, y_p2] += dS * (
                S[x, y_m2] * S[x, y_m1] * S[x, y_p1]
                + S[x, y_m1] * S[x, y_p1] * S[x, y_p3]
                + S[x, y_p1] * S[x, y_p3] * S[x, y_p4]
            )
            dE[x, y_p3] += dS * (
                S[x, y_m1] * S[x, y_p1] * S[x, y_p2]
                + S[x, y_p1] * S[x, y_p2] * S[x, y_p4]
            )
            dE[x, y_p4] += dS * S[x, y_p1] * S[x, y_p2] * S[x, y_p3]
        else:
            raise ValueError("p must be 2, 3, 4, or 5")
