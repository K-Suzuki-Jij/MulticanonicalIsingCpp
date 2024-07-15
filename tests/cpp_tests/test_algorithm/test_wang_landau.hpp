#pragma once

#include "../../../include/algorithm/wang_landau.hpp"
#include "gtest/gtest.h"

namespace cpp_muca {
namespace test {

TEST(AlgorithmWangLandau, BaseWangLandauResults) {
   const BaseWangLandauResults result({{0, 0.0}, {1, 0.1}, {2, 0.2}},
                                      OrderParameters(), 100, 0.5);

   EXPECT_EQ(result.entropy_dict.at(0), 0.0);
   EXPECT_EQ(result.entropy_dict.at(1), 0.1);
   EXPECT_EQ(result.entropy_dict.at(2), 0.2);
   EXPECT_EQ(result.order_parameters.mag_2.size(), 0);
   EXPECT_EQ(result.order_parameters.mag_4.size(), 0);
   EXPECT_EQ(result.order_parameters.abs_f2.size(), 0);
   EXPECT_EQ(result.order_parameters.abs_f4.size(), 0);
   EXPECT_EQ(result.order_parameters.normalized_energy_count.size(), 0);
   EXPECT_EQ(result.total_sweeps, 100);
   EXPECT_EQ(result.final_modification_factor, 0.5);

}

TEST(AlgorithmWangLandau, BaseWangLandauMETROPOLIS) {
   const PBodyTwoDimIsing model(-1.0, 3, 6, 6, 1, 1);
   const WangLandauParameters wl_parameters(1e-02, 100, 1, 0, std::numeric_limits<std::int64_t>::max(), 0.8, 0.5, 0.2, UpdateMethod::METROPOLIS);
   const auto result = BaseWangLandau(model, wl_parameters, 0, {-2*6*6*2*2*2, 0.0}, true);
   
   EXPECT_TRUE(result.final_modification_factor < 1e-02);
   EXPECT_EQ(result.order_parameters.mag_2.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.mag_4.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f2.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f4.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.normalized_energy_count.size(), result.entropy_dict.size());
   
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   
   EXPECT_EQ(hist_min, -2*6*6*2*2*2);
   EXPECT_EQ(hist_max, 0);
   for (const auto &it: result.entropy_dict) {
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
}

TEST(AlgorithmWangLandau, BaseWangLandauHEATBATH) {
   const PBodyTwoDimIsing model(-1.0, 3, 6, 6, 1, 1);
   const WangLandauParameters wl_parameters(1e-02, 100, 1, 0, std::numeric_limits<std::int64_t>::max(), 0.8, 0.5, 0.2, UpdateMethod::HEAT_BATH);
   const auto result = BaseWangLandau(model, wl_parameters, 0, {-2*6*6*2*2*2, 0.0}, true);
   
   EXPECT_TRUE(result.final_modification_factor < 1e-02);
   EXPECT_EQ(result.order_parameters.mag_2.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.mag_4.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f2.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.abs_f4.size(), result.entropy_dict.size());
   EXPECT_EQ(result.order_parameters.normalized_energy_count.size(), result.entropy_dict.size());
   
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   
   EXPECT_EQ(hist_min, -2*6*6*2*2*2);
   EXPECT_EQ(hist_max, 0);
   for (const auto &it: result.entropy_dict) {
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
}

TEST(AlgorithmWangLandau, BaseWangLandauError) {
   const PBodyTwoDimIsing model(-1.0, 3, 6, 6, 1, 1);
   const WangLandauParameters wl_parameters(1e-02, 100, 1, 0, 2);
   EXPECT_THROW(BaseWangLandau(model, wl_parameters, 0, {-2*6*6*2*2*2, 0.0}, true), std::runtime_error);
}

TEST(AlgorithmWangLandau, WangLandau) {
   const PBodyTwoDimIsing model(-2, 2, 4, 4, 0.5, 2);
   const WangLandauParameters wl_parameters(1e-08, 100, 2, 0, 100000, 0.8, 0.5, 0.2);
   const auto result_list = WangLandau(model, wl_parameters, 2, true);
   
   EXPECT_EQ(result_list.size(), 2);
   
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result_list[0].entropy_dict.begin(),
                                         result_list[0].entropy_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result_list[1].entropy_dict.begin(),
                                         result_list[1].entropy_dict.end(),
                                         min_key)->first;
   EXPECT_EQ(hist_min, -2*4*4);
   EXPECT_EQ(hist_max, +2*4*4);
}

TEST(AlgorithmWangLandau, WangLandauSymmetric1) {
   const PBodyTwoDimIsing model(-2, 2, 4, 4, 0.5, 2);
   const WangLandauParameters wl_parameters(1e-08, 100, 1, 1, 100000, 0.8, 0.5, 0.2);
   const auto result = WangLandauSymmetric(model, wl_parameters);
      
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   EXPECT_EQ(hist_min, -2*4*4);
   EXPECT_EQ(hist_max, +2*4*4);
}

TEST(AlgorithmWangLandau, WangLandauSymmetric2) {
   const PBodyTwoDimIsing model(-2, 3, 6, 6, 0.5, 2);
   const WangLandauParameters wl_parameters(1e-08, 100, 1, 0, 100000, 0.8, 0.5, 0.2);
   const auto result = WangLandauSymmetric(model, wl_parameters);
      
   const auto min_key = [](const auto &a, const auto &b) {return a.first < b.first;};
   const int hist_min = std::min_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   const int hist_max = std::max_element(result.entropy_dict.begin(),
                                         result.entropy_dict.end(),
                                         min_key)->first;
   EXPECT_EQ(hist_min, -2*6*6);
   EXPECT_EQ(hist_max, +2*6*6);
}

};
};
