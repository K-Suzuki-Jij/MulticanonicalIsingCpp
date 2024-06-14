#pragma once

#include "../model/p_body_ising.hpp"
#include <unordered_map>
#include <vector>
#include <complex>
#include <random>


namespace cpp_muca {

struct OrderParameters {
   std::unordered_map<int, double> mag_2;
   std::unordered_map<int, double> mag_4;
   std::unordered_map<int, double> abs_f2;
   std::unordered_map<int, double> abs_f4;
   std::unordered_map<int, std::int64_t> normalized_energy_count;
};

class OrderParameterCounter {

public:
   OrderParameterCounter(const PBodyTwoDimIsing &model, const std::vector<std::vector<int>> &initial_spins):
   model(model) {
      this->op_coeff = 0.5*model.spin_scale_factor/(model.Lx*model.Ly);
      this->fourier = CalculateFourier(initial_spins);
      this->order_parameters = OrderParameters();

      this->ordered_Q.reserve(model.p*model.p);
      for (int i = 0; i < model.p; ++i) {
         for (int j = 0; j < model.p; ++j) {
            this->ordered_Q.push_back({i*model.Lx/model.p, j*model.Ly/model.p});
         }
      }
   }

   void UpdateFourier(const double dS, const int x, const int y) {
      for (const auto &[qx, qy]: this->ordered_Q) {
         const double img = -2.0*M_PI*(static_cast<double>(x*qx)/model.Lx + static_cast<double>(y*qy)/model.Ly);
         this->fourier[qx][qy] += this->op_coeff*dS*std::exp(std::complex<double>(0.0, img));
      }
   }

   void UpdateOrderParameters(const int normalized_energy) {
      double temp_abs_f2 = 0.0;
      for (const auto &[qx, qy]: this->ordered_Q) {
         temp_abs_f2 += std::norm(this->fourier[qx][qy]);
      }
      const double sq_mag = std::norm(this->fourier[0][0]);
      this->order_parameters.abs_f2[normalized_energy] += temp_abs_f2;
      this->order_parameters.abs_f4[normalized_energy] += temp_abs_f2*temp_abs_f2;
      this->order_parameters.mag_2[normalized_energy] += sq_mag;
      this->order_parameters.mag_4[normalized_energy] += sq_mag*sq_mag;
      this->order_parameters.normalized_energy_count[normalized_energy] += 1;
   }
   
   void UpdateOrderParametersAt(const int normalized_energy) {
      double temp_abs_f2 = 0.0;
      for (const auto &[qx, qy]: this->ordered_Q) {
         temp_abs_f2 += std::norm(this->fourier[qx][qy]);
      }
      const double sq_mag = std::norm(this->fourier[0][0]);
      this->order_parameters.abs_f2.at(normalized_energy) += temp_abs_f2;
      this->order_parameters.abs_f4.at(normalized_energy) += temp_abs_f2*temp_abs_f2;
      this->order_parameters.mag_2.at(normalized_energy) += sq_mag;
      this->order_parameters.mag_4.at(normalized_energy) += sq_mag*sq_mag;
      this->order_parameters.normalized_energy_count.at(normalized_energy) += 1;
   }
   
   void ReserveEnergyKyes(const std::unordered_map<int, double> &entropy_dict,
                          const std::pair<double, double> &normalized_energy_range) {
      for (const auto &it: entropy_dict) {
         if (normalized_energy_range.first <= it.first && it.first <= normalized_energy_range.second) {
            this->order_parameters.abs_f2[it.first] = 0;
            this->order_parameters.abs_f4[it.first] = 0;
            this->order_parameters.mag_2[it.first] = 0;
            this->order_parameters.mag_4[it.first] = 0;
            this->order_parameters.normalized_energy_count[it.first] = 0;
         }
      }
   }

   const OrderParameters &ToOrderParameters() const {
      return this->order_parameters;
   }

   const PBodyTwoDimIsing &GetModel() const {
      return this->model;
   }

   const double GetOpCoeff() const {
      return this->op_coeff;
   }

   const std::vector<std::pair<int, int>> &GetOrderedQ() const {
      return this->ordered_Q;
   }

   const std::vector<std::vector<std::complex<double>>> &GetFourier() const {
      return this->fourier;
   }

private:
   const PBodyTwoDimIsing model;
   double op_coeff;
   OrderParameters order_parameters;
   std::vector<std::pair<int, int>> ordered_Q;
   std::vector<std::vector<std::complex<double>>> fourier;

   std::vector<std::vector<std::complex<double>>> CalculateFourier(const std::vector<std::vector<int>> &initial_spins) const {
      std::vector<std::vector<std::complex<double>>> fourier(model.Lx, std::vector<std::complex<double>>(model.Ly));
      for (int i = 0; i < model.Lx; ++i) {
         for (int j = 0; j < model.Ly; ++j) {
            auto temp = std::complex<double>(0.0, 0.0);
            for (int k = 0; k < model.Lx; ++k) {
               for (int l = 0; l < model.Ly; ++l) {
                  const double img = -2.0*M_PI*(static_cast<double>(i*k)/model.Lx + static_cast<double>(j*l)/model.Ly);
                  temp += std::exp(std::complex<double>(0.0, img))*static_cast<double>(initial_spins[k][l]);
               }
            }
            fourier[i][j] = this->op_coeff*temp;
         }
      }
      return fourier;
   }
};

std::vector<std::vector<int>> 
GenerateInitialState(const PBodyTwoDimIsing &model, 
                     const std::pair<double, double> &normalized_energy_range,
                     const std::size_t seed,
                     const std::int64_t max_trial = 1000000) {
   const double e_min = normalized_energy_range.first;
   const double e_max = normalized_energy_range.second;
   const int twice_spin_magnitude = static_cast<int>(2*model.spin);
   
   std::unordered_map<int, std::int64_t> entropy_dict;
   std::mt19937_64 engine(seed);
   std::uniform_real_distribution<double> ud(0.0, 1.0);
   std::uniform_int_distribution<int> dist_all_spins(0, twice_spin_magnitude);
   std::uniform_int_distribution<int> dist_new_spins(0, twice_spin_magnitude - 1);
   
   std::vector<std::vector<int>> twice_spins(model.Lx, std::vector<int>(model.Ly, 0));
   for (int x = 0; x < model.Lx; ++x) {
      for (int y = 0; y < model.Ly; ++y) {
         twice_spins[x][y] = 2*dist_all_spins(engine) - twice_spin_magnitude;
      }
   }
   std::vector<std::vector<int>> dE = model.MakeEnergyDifference(twice_spins);
   int normalized_energy = model.CalculateNormalizedEnergy(twice_spins);

   for (int trial = 0; trial < max_trial; ++trial) {
      for (int x = 0; x < model.Lx; ++x) {
         for (int y = 0; y < model.Ly; ++y) {
            int new_spin_value = 2*dist_new_spins(engine) - twice_spin_magnitude;
            if (twice_spins[x][y] <= new_spin_value) {
               new_spin_value += 2;
            }
            const int new_normalized_energy = normalized_energy + dE[x][y]*(new_spin_value - twice_spins[x][y]);
            // If the new state has the desired energy, return it
            if (e_min <= new_normalized_energy && new_normalized_energy <= e_max) {
               twice_spins[x][y] = new_spin_value;
               return twice_spins;
            }

            // Do the Wang-Landau update
            const double dS = entropy_dict[new_normalized_energy] - entropy_dict[normalized_energy];
            if (dS <= 0.0 || ud(engine) < std::exp(-dS)){
               model.UpdateEnergyDifference(dE, new_spin_value, x, y, twice_spins);
               twice_spins[x][y] = new_spin_value;
               normalized_energy = new_normalized_energy;
            }
            entropy_dict[normalized_energy] += 1;
         }
      }
   }

   throw std::runtime_error("Failed to generate an initial state with the desired energy range.");
}

 std::vector<std::pair<double, double>> 
 GenerateEnergyRangeList(const std::pair<int, int> &normalized_energy_range,
                         const int num_divided_energy_range,
                         const double overlap_rate) {
   const int e_min = normalized_energy_range.first;
   const int e_max = normalized_energy_range.second;
   const double length = (e_max - e_min)/(num_divided_energy_range - overlap_rate*(num_divided_energy_range - 1));
   std::vector<std::pair<double, double>> divided_energy_range_list;
   for (int i = 0; i < num_divided_energy_range; ++i) {
      const double e_left = e_min + i*length*(1 - overlap_rate);
      const double e_right = e_left + length;
      divided_energy_range_list.push_back(std::make_pair(e_left, e_right));
   }
   return divided_energy_range_list;
 }






} // namespace cpp_muca
