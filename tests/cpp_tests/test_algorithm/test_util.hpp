#pragma once

#include "../../../include/algorithm/util.hpp"
#include "../../../include/model/p_body_ising.hpp"
#include "gtest/gtest.h"

namespace cpp_muca {
namespace test {

TEST(AlgorithmUtil, OrderParameters) {

   auto op = OrderParameters();
   op.abs_f2[0] = 1.0;
   op.abs_f4[0] = 2.0;
   op.mag_2[0] = 3.0;
   op.mag_4[0] = 4.0;
   op.normalized_energy_count[0] = 4;

   EXPECT_EQ(op.abs_f2.at(0), 1.0);
   EXPECT_EQ(op.abs_f4.at(0), 2.0);
   EXPECT_EQ(op.mag_2.at(0), 3.0);
   EXPECT_EQ(op.mag_4.at(0), 4.0);
   EXPECT_EQ(op.normalized_energy_count.at(0), 4);
}


TEST(AlgorithmUtil, OrderParameterCounter) {
   PBodyTwoDimIsing model(1.0, 2, 4, 4, 0.5, 2);
   std::vector<std::vector<int>> initial_spins = {
      {+1, -1, +1, -1},
      {-1, +1, -1, +1},
      {+1, -1, +1, -1},
      {-1, +1, -1, +1}
   };

   auto opc = OrderParameterCounter(model, initial_spins);
   EXPECT_EQ(opc.ToOrderParameters().mag_2.size(), 0);
   EXPECT_EQ(opc.ToOrderParameters().mag_4.size(), 0);
   EXPECT_EQ(opc.ToOrderParameters().abs_f2.size(), 0);
   EXPECT_EQ(opc.ToOrderParameters().abs_f4.size(), 0);
   EXPECT_EQ(opc.ToOrderParameters().normalized_energy_count.size(), 0);
   
   EXPECT_THROW(opc.UpdateOrderParametersAt(99), std::out_of_range);

   opc.UpdateOrderParameters(99);
   EXPECT_EQ(opc.ToOrderParameters().mag_2.at(99), 0);
   EXPECT_EQ(opc.ToOrderParameters().mag_4.at(99), 0);
   EXPECT_EQ(opc.ToOrderParameters().abs_f2.at(99), 1);
   EXPECT_EQ(opc.ToOrderParameters().abs_f4.at(99), 1);
   EXPECT_EQ(opc.ToOrderParameters().normalized_energy_count.at(99), 1);
   
   EXPECT_THROW(opc.UpdateOrderParametersAt(100), std::out_of_range);
   opc.ReserveEnergyKyes(std::unordered_map<int, double>{{100, -1}}, std::pair<double, double>{99, 100});
   EXPECT_NO_THROW(opc.UpdateOrderParametersAt(100));
   
   EXPECT_EQ(opc.GetModel().J                , 1.0);
   EXPECT_EQ(opc.GetModel().p                , 2  );
   EXPECT_EQ(opc.GetModel().Lx               , 4  );
   EXPECT_EQ(opc.GetModel().Ly               , 4  );
   EXPECT_EQ(opc.GetModel().spin             , 0.5);
   EXPECT_EQ(opc.GetModel().spin_scale_factor, 2  );

   EXPECT_EQ(opc.GetOpCoeff(), 0.5*2/16);
   EXPECT_EQ(opc.GetOrderedQ(), (std::vector<std::pair<int, int>>{
      {0, 0}, {0, 2}, {2, 0}, {2, 2} 
   }));

   EXPECT_NEAR(opc.GetFourier().at(0).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(1).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(2).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(2).real(), 1.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(3).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(3).imag(), 0.0, 1e-15);
   
   opc.UpdateFourier(2, 0, 1);
   opc.UpdateFourier(2, 0, 3);
   opc.UpdateFourier(2, 1, 0);
   opc.UpdateFourier(2, 1, 2);
   opc.UpdateFourier(2, 2, 1);
   opc.UpdateFourier(2, 2, 3);
   opc.UpdateFourier(2, 3, 0);
   opc.UpdateFourier(2, 3, 2);
   
   EXPECT_NEAR(opc.GetFourier().at(0).at(0).real(), 1.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(0).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(1).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(1).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(2).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(2).at(3).imag(), 0.0, 1e-15);
   
   EXPECT_NEAR(opc.GetFourier().at(3).at(0).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(0).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(1).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(1).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(2).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(2).imag(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(3).real(), 0.0, 1e-15);
   EXPECT_NEAR(opc.GetFourier().at(3).at(3).imag(), 0.0, 1e-15);
   
}

TEST(AlgorithmUtil, GenerateInitialState) {
   PBodyTwoDimIsing model(-1.0, 2, 4, 4, 0.5, 2);
   const auto spins = GenerateInitialState(model, std::pair<double, double>{-32.0, -31.0}, 0, 1000);
   EXPECT_EQ(model.CalculateNormalizedEnergy(spins), -32.0);
   EXPECT_EQ(spins.size(), 4);
   EXPECT_EQ(spins[0].size(), 4);
   EXPECT_EQ(spins[1].size(), 4);
   EXPECT_EQ(spins[2].size(), 4);
   EXPECT_EQ(spins[3].size(), 4);

   EXPECT_EQ(spins[0][0], -1);
   EXPECT_EQ(spins[0][1], -1);
   EXPECT_EQ(spins[0][2], -1);
   EXPECT_EQ(spins[0][3], -1);
   EXPECT_EQ(spins[1][0], -1);
   EXPECT_EQ(spins[1][1], -1);
   EXPECT_EQ(spins[1][2], -1);
   EXPECT_EQ(spins[1][3], -1);
   EXPECT_EQ(spins[2][0], -1);
   EXPECT_EQ(spins[2][1], -1);
   EXPECT_EQ(spins[2][2], -1);
   EXPECT_EQ(spins[2][3], -1);
   EXPECT_EQ(spins[3][0], -1);
   EXPECT_EQ(spins[3][1], -1);
   EXPECT_EQ(spins[3][2], -1);
   EXPECT_EQ(spins[3][3], -1);
   
   EXPECT_THROW(GenerateInitialState(model, std::pair<double, double>{-33.0, -32.1}, 0, 100), std::runtime_error);
}


TEST(AlgorithmUtil, GenerateEnergyRangeList) {
   const auto result_1 = GenerateEnergyRangeList(std::pair<int, int>{-10, 10}, 2, 0);
   EXPECT_EQ(result_1[0].first,  -10);
   EXPECT_EQ(result_1[0].second, 0  );
   EXPECT_EQ(result_1[1].first,  0  );
   EXPECT_EQ(result_1[1].second, 10 );
   
   const auto result_2 = GenerateEnergyRangeList(std::pair<int, int>{-10, 10}, 3, 0);
   EXPECT_NEAR(result_2[0].first,  -10.0  , 1e-10);
   EXPECT_NEAR(result_2[0].second, -10.0/3, 1e-10);
   EXPECT_NEAR(result_2[1].first,  -10.0/3, 1e-10);
   EXPECT_NEAR(result_2[1].second, 10.0/3 , 1e-10);
   EXPECT_NEAR(result_2[2].first,  10.0/3 , 1e-10);
   EXPECT_NEAR(result_2[2].second, 10.0   , 1e-10);
   
   const auto result_3 = GenerateEnergyRangeList(std::pair<int, int>{-10, 10}, 2, 0.4);
   EXPECT_DOUBLE_EQ(result_3[0].first,  -10 );
   EXPECT_DOUBLE_EQ(result_3[0].second,  2.5);
   EXPECT_DOUBLE_EQ(result_3[1].first,  -2.5);
   EXPECT_DOUBLE_EQ(result_3[1].second,  10 );
   
   const auto result_4 = GenerateEnergyRangeList(std::pair<int, int>{-10, 10}, 3, 0.4);
   EXPECT_NEAR(result_4[0].first,  -10             , 1e-10);
   EXPECT_NEAR(result_4[0].second, -10 + 20/2.2    , 1e-10);
   EXPECT_NEAR(result_4[1].first,  -10 + 0.6*20/2.2, 1e-10);
   EXPECT_NEAR(result_4[1].second, -10 + 1.6*20/2.2, 1e-10);
   EXPECT_NEAR(result_4[2].first,  -10 + 120.0/11  , 1e-10);
   EXPECT_NEAR(result_4[2].second, -10 + 2.2*20/2.2, 1e-10);
}

};
};
