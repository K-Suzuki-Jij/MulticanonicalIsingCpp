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
                                     const bool calculate_order_parameters
                                     ) {

   std::mt19937_64 engine(seed);
   const double e_min = normalized_energy_range.first;
   const double e_max = normalized_energy_range.second;
   const int twice_spin_magnitude = static_cast<int>(2*model.spin);
   std::vector<std::vector<int>> twice_spins = GenerateInitialState(model, normalized_energy_range, engine());
   std::vector<std::vector<int>> dE = model.MakeEnergyDifference(twice_spins);
   std::vector<double> prob_list(twice_spin_magnitude + 1, 1.0);
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

      if (parameters.update_method == UpdateMethod::METROPOLIS) {
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
      }
      else if (parameters.update_method == UpdateMethod::HEAT_BATH) {
         for (int x = 0; x < model.Lx; ++x) {
            for (int y = 0; y < model.Ly; ++y) {
               // Calculate probability
               double z = 0.0;
               for (int state = 0; state <= twice_spin_magnitude; ++state) {
                  const int new_spin_value = 2*state - twice_spin_magnitude;
                  const int new_normalized_energy = normalized_energy + dE[x][y]*(new_spin_value - twice_spins[x][y]);
                  if (e_min <= new_normalized_energy && new_normalized_energy <= e_max) {
                     const double dS = entropy_dict[new_normalized_energy] - entropy_dict[normalized_energy];
                     prob_list[state] = std::exp(-dS);
                  }
                  else {
                     prob_list[state] = 0.0;
                  }
                  z += prob_list[state];
               }

               if (z == 0.0) {
                  throw std::runtime_error("Probability is zero.");
               }
               // Select new spin value
               z = 1.0/z;
               double prob_sum = 0.0;
               const double dist = ud(engine);
               int selected_state = -1;
               for (std::int32_t state = 0; state <= twice_spin_magnitude; ++state) {
                  prob_sum += z*prob_list[state];
                  if (dist < prob_sum) {
                     selected_state = state;
                     break;
                  }
               }

               // Check if the selected state is valid
               if (prob_list[selected_state] == 0.0) {
                  throw std::runtime_error("Invalid state selected.");
               }

               const int new_spin_value = 2*selected_state - twice_spin_magnitude;
               const int new_normalized_energy = normalized_energy + dE[x][y]*(new_spin_value - twice_spins[x][y]);
               if (new_normalized_energy != normalized_energy) {
                  if (calculate_order_parameters) {
                     order_parameter_counter.UpdateFourier(new_spin_value - twice_spins[x][y], x, y);
                  }
                  model.UpdateEnergyDifference(dE, new_spin_value, x, y, twice_spins);
                  twice_spins[x][y] = new_spin_value;
                  normalized_energy = new_normalized_energy;
               }
               entropy_dict[normalized_energy] += diff;
               histogram_dict[normalized_energy] += 1;

               if (calculate_order_parameters) {
                  order_parameter_counter.UpdateOrderParameters(normalized_energy);
               }
            }
         }
      }
      else {
         throw std::invalid_argument("Invalid update method.");
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
      throw std::runtime_error("Dose not converge.");
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

BaseWangLandauResults WangLandauSymmetric(const PBodyTwoDimIsing &model, 
                                          const WangLandauParameters parameters) {

   // Check if the parameters are valid.
   if (parameters.num_divided_energy_range != 1) {
      throw std::invalid_argument("num_divided_energy_range must be 1.");
   }

   if (parameters.update_method != UpdateMethod::METROPOLIS) {
      throw std::invalid_argument("update_method must be METROPOLIS.");
   }

   // Define variables
   std::mt19937_64 engine(parameters.seed);
   std::unordered_map<int, std::int64_t> histogram_dict;
   std::unordered_map<int, double> entropy_dict;
   double diff = 1.0;
   bool loop_breaker = false;
   const int twice_spin_magnitude = static_cast<int>(2*model.spin);
   std::uniform_real_distribution<double> ud(0.0, 1.0);
   std::uniform_int_distribution<int> dist_new_spins(0, twice_spin_magnitude - 1);
   std::uniform_int_distribution<int> dist_all_spins(0, twice_spin_magnitude);
   std::int64_t total_sweep = 0;

   // Generate initial state
   std::vector<std::vector<int>> twice_spins(model.Lx, std::vector<int>(model.Ly, 0));
   for (int x = 0; x < model.Lx; ++x) {
      for (int y = 0; y < model.Ly; ++y) {
         twice_spins[x][y] = 2*dist_all_spins(engine) - twice_spin_magnitude;
      }
   }

   // Calculate energy difference
   std::vector<std::vector<int>> dE = model.MakeEnergyDifference(twice_spins);
   int normalized_energy = model.CalculateNormalizedEnergy(twice_spins);

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
            //const double dS = entropy_dict[new_normalized_energy] - entropy_dict[normalized_energy];
            const double dS = entropy_dict[std::abs(new_normalized_energy)] - entropy_dict[std::abs(normalized_energy)];
            if (dS <= 0.0 || ud(engine) < std::exp(-dS)) {
               model.UpdateEnergyDifference(dE, new_spin_value, x, y, twice_spins);
               twice_spins[x][y] = new_spin_value;
               normalized_energy = new_normalized_energy;
            }
            entropy_dict[std::abs(normalized_energy)] += diff;
            histogram_dict[std::abs(normalized_energy)] += 1;
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
      throw std::runtime_error("Dose not converge.");
   }

   // Extract keys of entropy_dict
   std::vector<int> keys;
   for (const auto &it: entropy_dict) {
      keys.push_back(it.first);
   }

   // Fill minus keys
   for (const int it: keys) { 
      entropy_dict[-it] = entropy_dict.at(it);
   }

   return BaseWangLandauResults(entropy_dict, OrderParameters(), total_sweep + 1, diff/parameters.reduce_rate);
};




} // namespace cpp_muca
