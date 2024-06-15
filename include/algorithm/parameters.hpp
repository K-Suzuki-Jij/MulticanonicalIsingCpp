#pragma once

#include <stdexcept>
#include <limits>

namespace cpp_muca {

struct WangLandauParameters {

   WangLandauParameters(const double modification_criterion,
                        const int convergence_check_interval,
                        const int num_divided_energy_range,
                        const std::size_t seed,
                        const std::int64_t max_sweeps = std::numeric_limits<std::int64_t>::max(),
                        const double flatness_criterion = 0.8,
                        const double reduce_rate = 0.5,
                        const double overlap_rate = 0.2):
   modification_criterion(modification_criterion),
   convergence_check_interval(convergence_check_interval),
   num_divided_energy_range(num_divided_energy_range),
   seed(seed),
   max_sweeps(max_sweeps),
   flatness_criterion(flatness_criterion),
   reduce_rate(reduce_rate),   
   overlap_rate(overlap_rate) {
      if (this->modification_criterion <= 0.0 || this->modification_criterion > 1.0) {
         throw std::invalid_argument("modification_criterion must be in the range (0, 1].");
      }
      if (this->convergence_check_interval <= 0) {
         throw std::invalid_argument("convergence_check_interval must be positive.");
      }
      if (this->flatness_criterion < 0.0 || this->flatness_criterion >= 1.0) {
         throw std::invalid_argument("flatness_criterion must be in the range [0, 1).");
      }
      if (this->reduce_rate <= 0.0 || this->reduce_rate >= 1.0) {
         throw std::invalid_argument("reduce_rate must be in the range (0, 1).");
      }
      if (this->max_sweeps <= 0) {
         throw std::invalid_argument("max_sweeps must be positive.");
      }
      if (this->num_divided_energy_range <= 0) {
         throw std::invalid_argument("num_divided_energy_range must be positive.");
      }
      if (this->overlap_rate < 0.0 || this->overlap_rate > 1.0) {
         throw std::invalid_argument("overlap_rate must be in the range [0, 1].");
      }
   }

   const double modification_criterion = 1e-08;
   const int convergence_check_interval = 1000;
   const int num_divided_energy_range = 1;
   const std::size_t seed = 0;
   const std::int64_t max_sweeps = 100000000;
   const double flatness_criterion = 0.8;
   const double reduce_rate = 0.5;
   const double overlap_rate = 0.2;
};

struct MulticanonicalParameters {


   MulticanonicalParameters(const std::int64_t num_sweeps,
                           const std::size_t seed,
                           const int num_divided_energy_range,
                           const double overlap_rate):
   num_sweeps(num_sweeps), 
   seed(seed),
   num_divided_energy_range(num_divided_energy_range),
   overlap_rate(overlap_rate) {
      if (this->num_sweeps <= 0) {
         throw std::invalid_argument("num_sweeps must be positive.");
      }
      if (this->num_divided_energy_range <= 0) {
         throw std::invalid_argument("num_divided_energy_range must be positive.");
      }
      if (this->overlap_rate < 0.0 || this->overlap_rate > 1.0) {
         throw std::invalid_argument("overlap_rate must be in the range [0, 1].");
      }

   }

   const std::int64_t num_sweeps = 1000;
   const std::size_t seed = 0;
   const int num_divided_energy_range = 1;
   const double overlap_rate = 0.2; 
};

} // namespace cpp_muca
