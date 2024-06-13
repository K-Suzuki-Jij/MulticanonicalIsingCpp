#pragma once

#include "util.hpp"
#include "parameters.hpp"
#include "../model/p_body_ising.hpp"
#include <random>
#include <iomanip>

namespace cpp_muca {

struct BaseMulticanonicalResults {

   BaseMulticanonicalResults(const std::unordered_map<int, std::int64_t> histogram_dict = {},
                             const OrderParameters order_parameters = OrderParameters()):
   histogram_dict(histogram_dict), order_parameters(order_parameters) {

   }

   std::unordered_map<int, std::int64_t> histogram_dict;
   OrderParameters order_parameters;
};


BaseMulticanonicalResults BaseMulticanonical(const PBodyTwoDimIsing model,
                                             const std::size_t seed,
                                             const std::int64_t num_sweeps,
                                             const std::unordered_map<int, double> entropy_dict,
                                             const std::pair<double, double> normalized_energy_range,
                                             const bool calculate_order_parameters) {

   std::mt19937_64 engine(seed);
   const double e_min = normalized_energy_range.first;
   const double e_max = normalized_energy_range.second;
   const int twice_spin_magnitude = static_cast<int>(2*model.spin);
   std::vector<std::vector<int>> twice_spins = GenerateInitialState(model, normalized_energy_range, engine());
   std::vector<std::vector<int>> dE = model.MakeEnergyDifference(twice_spins);
   
   OrderParameterCounter order_parameter_counter = OrderParameterCounter(model, twice_spins);
   std::uniform_real_distribution<double> ud(0.0, 1.0);
   std::uniform_int_distribution<int> dist_new_spins(0, twice_spin_magnitude - 1);
   
   std::unordered_map<int, std::int64_t> histogram_dict;
   for (const auto &it: entropy_dict) {
      if (e_min <= it.first && it.first <= e_max) {
         histogram_dict[it.first] = 0;
      }
   }
   order_parameter_counter.ReserveEnergyKyes(entropy_dict, normalized_energy_range);
   
   int normalized_energy = model.CalculateNormalizedEnergy(twice_spins);
   
   for (std::int64_t sweep = 0; sweep < num_sweeps; ++sweep) {
      for (int x = 0; x < model.Lx; ++x) {
         for (int y = 0; y < model.Ly; ++y) {
            int new_spin_value = 2*dist_new_spins(engine) - twice_spin_magnitude;
            if (twice_spins[x][y] <= new_spin_value) {
               new_spin_value += 2;
            }
            const int new_normalized_energy = normalized_energy + dE[x][y]*(new_spin_value - twice_spins[x][y]);
            if (e_min <= new_normalized_energy && new_normalized_energy <= e_max) {
               const double dS = entropy_dict.at(new_normalized_energy) - entropy_dict.at(normalized_energy);
               if (dS <= 0.0 || ud(engine) < std::exp(-dS)) {
                  if (calculate_order_parameters) {
                     order_parameter_counter.UpdateFourier(new_spin_value - twice_spins[x][y], x, y);
                  }
                  model.UpdateEnergyDifference(dE, new_spin_value, x, y, twice_spins);
                  twice_spins[x][y] = new_spin_value;
                  normalized_energy = new_normalized_energy;
               }
            }
            histogram_dict.at(normalized_energy) += 1;
            if (calculate_order_parameters) {
               order_parameter_counter.UpdateOrderParametersAt(normalized_energy);
            }
         }
      }
   }

   return BaseMulticanonicalResults(histogram_dict, order_parameter_counter.ToOrderParameters());

};

std::vector<BaseMulticanonicalResults> 
Multicanonical(const PBodyTwoDimIsing &model, 
               const MulticanonicalParameters &parameters,
               const std::unordered_map<int, double> &entropy_dict,
               const int num_threads,
               const bool calculate_order_parameters) {

   // Get energy range
   const auto normalized_energy_range_list = GenerateEnergyRangeList(model.normalized_energy_range, 
                                                                     parameters.num_divided_energy_range, 
                                                                     parameters.overlap_rate);

   // Generate seed list;
   std::mt19937_64 engine(parameters.seed);
   std::vector<std::size_t> seed_list(parameters.num_divided_energy_range);
   for (int i = 0; i < parameters.num_divided_energy_range; ++i) {
      seed_list[i] = engine();
   }
   
   // Run multicanonical simulation
   std::vector<BaseMulticanonicalResults> result_list(parameters.num_divided_energy_range);
   
#pragma omp parallel for num_threads(num_threads)
   for (int i = 0; i < parameters.num_divided_energy_range; ++i) {
      result_list[i] = BaseMulticanonical(model, 
                                          seed_list[i],
                                          parameters.num_sweeps,
                                          entropy_dict,
                                          normalized_energy_range_list[i],
                                          calculate_order_parameters
                                          );
   }

   return result_list;
}


} // namespace cpp_muca
