#pragma once

#include "../../../include/algorithm/multicanonical.hpp"
#include "../../../include/algorithm/wang_landau.hpp"
#include "gtest/gtest.h"

namespace cpp_muca {
namespace test {

TEST(AlgorithmMulticanonical, BaseMulticanonicalResults) {

   std::unordered_map<int, std::int64_t> histogram_dict = {{0, 0}, {1, 1}, {2, 2}};
   OrderParameters order_parameters = OrderParameters();

   const BaseMulticanonicalResults result(
      histogram_dict, order_parameters
   );

   EXPECT_EQ(result.histogram_dict.at(0), 0);
   EXPECT_EQ(result.histogram_dict.at(1), 1);
   EXPECT_EQ(result.histogram_dict.at(2), 2);
   EXPECT_EQ(result.order_parameters.mag_2.size(), 0);
   EXPECT_EQ(result.order_parameters.mag_4.size(), 0);
   EXPECT_EQ(result.order_parameters.abs_f2.size(), 0);
   EXPECT_EQ(result.order_parameters.abs_f4.size(), 0);
   EXPECT_EQ(result.order_parameters.normalized_energy_count.size(), 0);

}

TEST(AlgorithmMulticanonical, BaseMulticanonical) {
   const PBodyTwoDimIsing model(-1, 3, 6, 6, 1, 1);
   const WangLandauParameters wl_parameters(1e-02, 100, 1, 0);
   const auto initial_data = BaseWangLandau(model, wl_parameters, 0, {-2*6*6*2*2*2, 0.0}, true);
   const MulticanonicalParameters parameters(1000, wl_parameters.seed, 1, 0.4);
   
   const auto result = BaseMulticanonical(model,
                                          parameters.seed,
                                          parameters.num_sweeps,
                                          initial_data.entropy_dict,
                                          {-2*6*6*2*2*2, 0.0},
                                          true);
   
   EXPECT_EQ(result.order_parameters.mag_2.size(), result.histogram_dict.size());
   EXPECT_EQ(result.order_parameters.mag_4.size(), result.histogram_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f2.size(), result.histogram_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f4.size(), result.histogram_dict.size());
   EXPECT_EQ(result.order_parameters.normalized_energy_count.size(), result.histogram_dict.size());
   
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result.histogram_dict.begin(), 
                                         result.histogram_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result.histogram_dict.begin(),
                                         result.histogram_dict.end(),
                                         min_key)->first;
   EXPECT_EQ(hist_min, -2*6*6*2*2*2);
   EXPECT_EQ(hist_max, 0);
   for (const auto &it: result.histogram_dict) {
      EXPECT_TRUE(it.second > 0);
      EXPECT_EQ(result.order_parameters.mag_2.count(it.first), 1);
      EXPECT_EQ(result.order_parameters.mag_4.count(it.first), 1);
      EXPECT_EQ(result.order_parameters.abs_f2.count(it.first), 1);
      EXPECT_EQ(result.order_parameters.abs_f4.count(it.first), 1);
      EXPECT_EQ(result.order_parameters.normalized_energy_count.count(it.first), 1);
   }
   for (const auto &it: result.order_parameters.mag_2) {
      EXPECT_TRUE(it.second > 0);
   }
   for (const auto &it: result.order_parameters.mag_4) {
      EXPECT_TRUE(it.second > 0);
   }
   for (const auto &it: result.order_parameters.abs_f2) {
      EXPECT_TRUE(it.second > 0);
   }
   for (const auto &it: result.order_parameters.abs_f4) {
      EXPECT_TRUE(it.second > 0);
   }
   for (const auto &it: result.order_parameters.normalized_energy_count) {
      EXPECT_TRUE(it.second > 1);
   }
   
   EXPECT_THROW(BaseMulticanonical(model,
                                   parameters.seed,
                                   parameters.num_sweeps,
                                   initial_data.entropy_dict,
                                   {0, 2*6*6*2*2*2},
                                   true), std::out_of_range);
}

TEST(AlgorithmMulticanonical, Multicanonical) {
   const PBodyTwoDimIsing model(-2, 2, 4, 4, 0.5, 2);
   const WangLandauParameters wl_parameters(1e-08, 100, 1, 0);
   const auto initial_data = BaseWangLandau(model, wl_parameters, 0, model.normalized_energy_range, true);
   const MulticanonicalParameters parameters(1000, wl_parameters.seed, 2, 0.4);
   const auto result_list = Multicanonical(model, parameters, initial_data.entropy_dict, 2, true);
   
   EXPECT_EQ(result_list.size(), 2);
   
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result_list[0].histogram_dict.begin(),
                                         result_list[0].histogram_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result_list[1].histogram_dict.begin(),
                                         result_list[1].histogram_dict.end(),
                                         min_key)->first;
   EXPECT_EQ(hist_min, -2*4*4);
   EXPECT_EQ(hist_max, +2*4*4);
}


};
};
