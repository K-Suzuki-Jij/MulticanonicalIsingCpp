#pragma once

#include "util.hpp"
#include "parameters.hpp"
#include "../model/p_body_ising.hpp"
#include <random>
#include <iomanip>

namespace cpp_muca {

struct BaseWangLandauResults {
   BaseWangLandauResults(const std::unordered_map<int, double> entropy_dict = {},
                         const OrderParameters order_parameters = OrderParameters(),
                         const std::int64_t total_sweeps = 0,
                         const double final_modification_factor = 0.0):
   entropy_dict(entropy_dict),
   order_parameters(order_parameters),
   total_sweeps(total_sweeps),
   final_modification_factor(final_modification_factor) {}

   std::unordered_map<int, double> entropy_dict;
   OrderParameters order_parameters;
   std::int64_t total_sweeps;
   double final_modification_factor;
};

BaseWangLandauResults BaseWangLandau(const PBodyTwoDimIsing model,
                                     const WangLandauParameters parameters,
                                     const std::size_t seed,
                                     const std::pair<double, double> normalized_energy_range,
                                     const bool calculate_order_parameters) {

   std::mt19937_64 engine(seed);
   const double e_min = normalized_energy_range.first;
   const double e_max = normalized_energy_range.second;
   const int twice_spin_magnitude = static_cast<int>(2*model.spin);
   std::vector<std::vector<int>> twice_spins = GenerateInitialState(model, normalized_energy_range, engine());
   std::vector<std::vector<int>> dE = model.MakeEnergyDifference(twice_spins);
   std::unordered_map<int, std::int64_t> histogram_dict;
   std::unordered_map<int, double> entropy_dict;
   double diff = 1.0;
   bool loop_breaker = false;
   int normalized_energy = model.CalculateNormalizedEnergy(twice_spins);
   OrderParameterCounter order_parameter_counter = OrderParameterCounter(model, twice_spins);
   std::uniform_real_distribution<double> ud(0.0, 1.0);
   std::uniform_int_distribution<int> dist_new_spins(0, twice_spin_magnitude - 1);
   std::int64_t total_sweep = 0;
   
   for (std::int64_t sweep = 0; sweep < parameters.max_sweeps; ++sweep) {
      if (loop_breaker) {
         total_sweep = sweep;
         break;
      }

      for (int x = 0; x < model.Lx; ++x) {
         for (int y = 0; y < model.Ly; ++y) {
            int new_spin_value = 2*dist_new_spins(engine) - twice_spin_magnitude;
            if (twice_spins[x][y] <= new_spin_value) {
               new_spin_value += 2;
            }
            const int new_normalized_energy = normalized_energy + dE[x][y]*(new_spin_value - twice_spins[x][y]);
            if (e_min <= new_normalized_energy && new_normalized_energy <= e_max) {
               const double dS = entropy_dict[new_normalized_energy] - entropy_dict[normalized_energy];
               if (dS <= 0.0 || ud(engine) < std::exp(-dS)) {
                  if (calculate_order_parameters) {
                     order_parameter_counter.UpdateFourier(new_spin_value - twice_spins[x][y], x, y);
                  }
                  model.UpdateEnergyDifference(dE, new_spin_value, x, y, twice_spins);
                  twice_spins[x][y] = new_spin_value;
                  normalized_energy = new_normalized_energy;
               }
            }
            entropy_dict[normalized_energy] += diff;
            histogram_dict[normalized_energy] += 1;

            if (calculate_order_parameters) {
               order_parameter_counter.UpdateOrderParameters(normalized_energy);
            }
         }
      }

      // Check if the histogram is flat
      if (sweep%parameters.convergence_check_interval == 0) {
         const double hist_min = std::min_element(histogram_dict.begin(), histogram_dict.end(), [](const auto &a, const auto &b) {
            return a.second < b.second;
         })->second;
         const double hist_mean = std::accumulate(histogram_dict.begin(), histogram_dict.end(), 0.0, [](const double &a, const auto &b) {
            return a + b.second;
         })/histogram_dict.size();

         if (hist_min > parameters.flatness_criterion*hist_mean) {
            if (diff < parameters.modification_criterion) {
               loop_breaker = true;
            }
            diff *= parameters.reduce_rate;
            histogram_dict.clear();
         }
      }
   }
   
   if (!loop_breaker) {
      std::runtime_error("Dose not converge.");
   }
   
   return BaseWangLandauResults(entropy_dict,
                                order_parameter_counter.ToOrderParameters(), 
                                total_sweep + 1, 
                                diff/parameters.reduce_rate);
};

std::vector<BaseWangLandauResults> 
WangLandau(const PBodyTwoDimIsing &model, 
           const WangLandauParameters parameters, 
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
   
   std::vector<BaseWangLandauResults> result_list(parameters.num_divided_energy_range);

#pragma omp parallel for num_threads(num_threads)
   for (int i = 0; i < parameters.num_divided_energy_range; ++i) {
      result_list[i] = BaseWangLandau(model, parameters, seed_list[i], normalized_energy_range_list[i], calculate_order_parameters);
   }

   return result_list;
}



} // namespace cpp_muca
