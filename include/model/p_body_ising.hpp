#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cpp_muca {

struct PBodyTwoDimIsing {
   
   PBodyTwoDimIsing(const double J,
                    const int p,
                    const int Lx,
                    const int Ly,
                    const double spin,
                    const double spin_scale_factor):
   J(J), p(p), Lx(Lx), Ly(Ly), spin(spin), spin_scale_factor(spin_scale_factor),
   normalized_energy_range(CalculateNormalizedEnergyRange(Lx, Ly, p, spin)) {
      if (this->J == 0.0) {
         throw std::invalid_argument("J must be non-zero.");
      }
      if (this->p < 2) {
         throw std::invalid_argument("p must be larger than 1.");
      }
      if (this->Lx <= 0) {
         throw std::invalid_argument("Lx must be positive.");
      }
      if (this->Ly <= 0) {
         throw std::invalid_argument("Ly must be positive.");
      }
      if (std::floor(2*this->spin) != 2*this->spin || this->spin < 0.5) {
         throw std::invalid_argument("spin must be positive half-integer.");
      }
      if (spin_scale_factor <= 0.0) {
         throw std::invalid_argument("spin_scale_factor must positive value");
      }
      if (Lx < p || Ly < p) {
         throw std::invalid_argument("Lx and Ly must be larger than p.");
      }
   }

   int CalculateNormalizedEnergy(const std::vector<std::vector<int>> &spin_configuration) const {
      int energy = 0;
      for (int x = 0; x < this->Lx; ++x) {
         for (int y = 0; y < this->Ly; ++y) {
            int prod_x = 1;
            int prod_y = 1;
            for (int i = 0; i < this->p; ++i) {
               prod_x *= spin_configuration[(x + i)%this->Lx][y];
               prod_y *= spin_configuration[x][(y + i)%this->Ly];
            }
            energy += prod_x + prod_y;
         }
      }
      return energy*((this->J > 0) - (this->J < 0));
   }

   std::vector<std::vector<int>> MakeEnergyDifference(const std::vector<std::vector<int>> &spin_configuration) const {
      const int sign = (this->J > 0) - (this->J < 0);
      auto energy_difference = std::vector<std::vector<int>>(this->Lx, std::vector<int>(this->Ly, 0));
      for (int x = 0; x < this->Lx; ++x) {
          for (int y = 0; y < this->Ly; ++y) {
            int val = 0;
            for (int i = 0; i < this->p; ++i) {
               int prod_x = 1;
               int prod_y = 1;
               for (int j = 0; j < this->p; ++j) {
                  const int px = (x - j + i + this->Lx)%this->Lx;
                  const int py = (y - j + i + this->Ly)%this->Ly;
                  if (px != x) {
                     prod_x *= spin_configuration[px][y];
                  }
                  if (py != y) {
                     prod_y *= spin_configuration[x][py];
                  }
               }
               val += prod_x + prod_y;
            }
            energy_difference[x][y] = sign*val;
         }
      }
      return energy_difference;
   }

   void UpdateEnergyDifference(std::vector<std::vector<int>> &energy_difference, 
                               const int new_spin_value, const int x, const int y, 
                               const std::vector<std::vector<int>> &spin_configuration
                               ) const {
      if (new_spin_value == spin_configuration[x][y]) {
         return;
      }
      const int sign = (this->J > 0) - (this->J < 0);
      if (this->p == 2) {
         Flip2Body(energy_difference, sign*(new_spin_value - spin_configuration[x][y]), x, y);
      }
      else if (this->p == 3) {
         Flip3Body(energy_difference, sign*(new_spin_value - spin_configuration[x][y]), spin_configuration, x, y);
      }
      else if (this->p == 4) {
         Flip4Body(energy_difference, sign*(new_spin_value - spin_configuration[x][y]), spin_configuration, x, y);
      }
      else if (this->p == 5) {
         Flip5Body(energy_difference, sign*(new_spin_value - spin_configuration[x][y]), spin_configuration, x, y);
      }
      else {
         throw std::invalid_argument("p must be 2, 3, 4, or 5.");
      }

   }

   void Flip2Body(std::vector<std::vector<int>> &dE, const int dS, const int x, const int y) const {
      dE[(x - 1 + this->Lx)%this->Lx][y] += dS;
      dE[(x + 1           )%this->Lx][y] += dS;
      dE[x][(y - 1 + this->Ly)%this->Ly] += dS;
      dE[x][(y + 1           )%this->Ly] += dS;
   }
   void Flip3Body(std::vector<std::vector<int>> &dE, const int dS, const std::vector<std::vector<int>> &S, const int x, const int y) const {
      const int x_m2 = (x - 2 + this->Lx)%this->Lx;
      const int x_m1 = (x - 1 + this->Lx)%this->Lx;
      const int x_p1 = (x + 1           )%this->Lx;
      const int x_p2 = (x + 2           )%this->Lx;
      const int y_m2 = (y - 2 + this->Ly)%this->Ly;
      const int y_m1 = (y - 1 + this->Ly)%this->Ly;
      const int y_p1 = (y + 1           )%this->Ly;
      const int y_p2 = (y + 2           )%this->Ly;
      dE[x_m2][y] += dS*(S[x_m1][y]);
      dE[x_m1][y] += dS*(S[x_m2][y] + S[x_p1][y]);
      dE[x_p1][y] += dS*(S[x_m1][y] + S[x_p2][y]);
      dE[x_p2][y] += dS*(S[x_p1][y]);
      dE[x][y_m2] += dS*(S[x][y_m1]);
      dE[x][y_m1] += dS*(S[x][y_m2] + S[x][y_p1]);
      dE[x][y_p1] += dS*(S[x][y_m1] + S[x][y_p2]);
      dE[x][y_p2] += dS*(S[x][y_p1]);
   }
   void Flip4Body(std::vector<std::vector<int>> &dE, const int dS, const std::vector<std::vector<int>> &S, const int x, const int y) const {
      const int x_m3 = (x - 3 + this->Lx)%this->Lx;
      const int x_m2 = (x - 2 + this->Lx)%this->Lx;
      const int x_m1 = (x - 1 + this->Lx)%this->Lx;
      const int x_p1 = (x + 1           )%this->Lx;
      const int x_p2 = (x + 2           )%this->Lx;
      const int x_p3 = (x + 3           )%this->Lx;
      const int y_m3 = (y - 3 + this->Ly)%this->Ly;
      const int y_m2 = (y - 2 + this->Ly)%this->Ly;
      const int y_m1 = (y - 1 + this->Ly)%this->Ly;
      const int y_p1 = (y + 1           )%this->Ly;
      const int y_p2 = (y + 2           )%this->Ly;
      const int y_p3 = (y + 3           )%this->Ly;
      dE[x_m3][y] += dS*(S[x_m2][y]*S[x_m1][y]);
      dE[x_m2][y] += dS*(S[x_m3][y]*S[x_m1][y] + S[x_m1][y]*S[x_p1][y]);
      dE[x_m1][y] += dS*(S[x_m3][y]*S[x_m2][y] + S[x_m2][y]*S[x_p1][y] + S[x_p1][y]*S[x_p2][y]);
      dE[x_p1][y] += dS*(S[x_m2][y]*S[x_m1][y] + S[x_m1][y]*S[x_p2][y] + S[x_p2][y]*S[x_p3][y]);
      dE[x_p2][y] += dS*(S[x_m1][y]*S[x_p1][y] + S[x_p1][y]*S[x_p3][y]);
      dE[x_p3][y] += dS*(S[x_p1][y]*S[x_p2][y]);
      dE[x][y_m3] += dS*(S[x][y_m2]*S[x][y_m1]);
      dE[x][y_m2] += dS*(S[x][y_m3]*S[x][y_m1] + S[x][y_m1]*S[x][y_p1]);
      dE[x][y_m1] += dS*(S[x][y_m3]*S[x][y_m2] + S[x][y_m2]*S[x][y_p1] + S[x][y_p1]*S[x][y_p2]);
      dE[x][y_p1] += dS*(S[x][y_m2]*S[x][y_m1] + S[x][y_m1]*S[x][y_p2] + S[x][y_p2]*S[x][y_p3]);
      dE[x][y_p2] += dS*(S[x][y_m1]*S[x][y_p1] + S[x][y_p1]*S[x][y_p3]);
      dE[x][y_p3] += dS*(S[x][y_p1]*S[x][y_p2]);
   }
   void Flip5Body(std::vector<std::vector<int>> &dE, const int dS, const std::vector<std::vector<int>> &S, const int x, const int y) const {
      const int x_m4 = (x - 4 + this->Lx)%this->Lx;
      const int x_m3 = (x - 3 + this->Lx)%this->Lx;
      const int x_m2 = (x - 2 + this->Lx)%this->Lx;
      const int x_m1 = (x - 1 + this->Lx)%this->Lx;
      const int x_p1 = (x + 1           )%this->Lx;
      const int x_p2 = (x + 2           )%this->Lx;
      const int x_p3 = (x + 3           )%this->Lx;
      const int x_p4 = (x + 4           )%this->Lx;
      const int y_m4 = (y - 4 + this->Ly)%this->Ly;
      const int y_m3 = (y - 3 + this->Ly)%this->Ly;
      const int y_m2 = (y - 2 + this->Ly)%this->Ly;
      const int y_m1 = (y - 1 + this->Ly)%this->Ly;
      const int y_p1 = (y + 1           )%this->Ly;
      const int y_p2 = (y + 2           )%this->Ly;
      const int y_p3 = (y + 3           )%this->Ly;
      const int y_p4 = (y + 4           )%this->Ly;
      dE[x_m4][y] += dS*(S[x_m3][y]*S[x_m2][y]*S[x_m1][y]);
      dE[x_m3][y] += dS*(S[x_m4][y]*S[x_m2][y]*S[x_m1][y] + S[x_m2][y]*S[x_m1][y]*S[x_p1][y]);
      dE[x_m2][y] += dS*(S[x_m4][y]*S[x_m3][y]*S[x_m1][y] + S[x_m3][y]*S[x_m1][y]*S[x_p1][y] + S[x_m1][y]*S[x_p1][y]*S[x_p2][y]);
      dE[x_m1][y] += dS*(S[x_m4][y]*S[x_m3][y]*S[x_m2][y] + S[x_m3][y]*S[x_m2][y]*S[x_p1][y] + S[x_m2][y]*S[x_p1][y]*S[x_p2][y] + S[x_p1][y]*S[x_p2][y]*S[x_p3][y]);
      dE[x_p1][y] += dS*(S[x_m3][y]*S[x_m2][y]*S[x_m1][y] + S[x_m2][y]*S[x_m1][y]*S[x_p2][y] + S[x_m1][y]*S[x_p2][y]*S[x_p3][y] + S[x_p2][y]*S[x_p3][y]*S[x_p4][y]);
      dE[x_p2][y] += dS*(S[x_m2][y]*S[x_m1][y]*S[x_p1][y] + S[x_m1][y]*S[x_p1][y]*S[x_p3][y] + S[x_p1][y]*S[x_p3][y]*S[x_p4][y]);
      dE[x_p3][y] += dS*(S[x_m1][y]*S[x_p1][y]*S[x_p2][y] + S[x_p1][y]*S[x_p2][y]*S[x_p4][y]);
      dE[x_p4][y] += dS*(S[x_p1][y]*S[x_p2][y]*S[x_p3][y]);
      dE[x][y_m4] += dS*(S[x][y_m3]*S[x][y_m2]*S[x][y_m1]);
      dE[x][y_m3] += dS*(S[x][y_m4]*S[x][y_m2]*S[x][y_m1] + S[x][y_m2]*S[x][y_m1]*S[x][y_p1]);
      dE[x][y_m2] += dS*(S[x][y_m4]*S[x][y_m3]*S[x][y_m1] + S[x][y_m3]*S[x][y_m1]*S[x][y_p1] + S[x][y_m1]*S[x][y_p1]*S[x][y_p2]);
      dE[x][y_m1] += dS*(S[x][y_m4]*S[x][y_m3]*S[x][y_m2] + S[x][y_m3]*S[x][y_m2]*S[x][y_p1] + S[x][y_m2]*S[x][y_p1]*S[x][y_p2] + S[x][y_p1]*S[x][y_p2]*S[x][y_p3]);
      dE[x][y_p1] += dS*(S[x][y_m3]*S[x][y_m2]*S[x][y_m1] + S[x][y_m2]*S[x][y_m1]*S[x][y_p2] + S[x][y_m1]*S[x][y_p2]*S[x][y_p3] + S[x][y_p2]*S[x][y_p3]*S[x][y_p4]);
      dE[x][y_p2] += dS*(S[x][y_m2]*S[x][y_m1]*S[x][y_p1] + S[x][y_m1]*S[x][y_p1]*S[x][y_p3] + S[x][y_p1]*S[x][y_p3]*S[x][y_p4]);
      dE[x][y_p3] += dS*(S[x][y_m1]*S[x][y_p1]*S[x][y_p2] + S[x][y_p1]*S[x][y_p2]*S[x][y_p4]);
      dE[x][y_p4] += dS*(S[x][y_p1]*S[x][y_p2]*S[x][y_p3]);
   }

   std::pair<double, double> CalculateNormalizedEnergyRange(const int Lx, const int Ly, const int p, const double spin) const {
      const int max_energy = 2*Lx*Ly*std::pow(static_cast<int>(2*spin), p);
      const int min_energy = -max_energy;
      return {min_energy, max_energy};
   }

   const double J;
   const int p;
   const int Lx;
   const int Ly;
   const double spin;
   const double spin_scale_factor;
   const std::pair<double, double> normalized_energy_range;
   
};
} // namespace cpp_muca
