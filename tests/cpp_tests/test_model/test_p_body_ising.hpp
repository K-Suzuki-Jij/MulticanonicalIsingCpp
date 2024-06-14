#pragma once

#include "../../../include/model/p_body_ising.hpp"
#include "gtest/gtest.h"

namespace cpp_muca {
namespace test {

TEST(AlgorithmModel, PBodyTwoDimIsing) {

   const double J = 1.0;
   const int p = 4;
   const int Lx = 8;
   const int Ly = 8;
   const double spin = 1.5;
   const double spin_scale_factor = 1.5;
   PBodyTwoDimIsing model(J, p, Lx, Ly, spin, spin_scale_factor);

   EXPECT_EQ(model.J, J);
   EXPECT_EQ(model.p, p);
   EXPECT_EQ(model.Lx, Lx);
   EXPECT_EQ(model.Ly, Ly);
   EXPECT_EQ(model.spin, spin);
   EXPECT_EQ(model.spin_scale_factor, spin_scale_factor);
   EXPECT_EQ(model.normalized_energy_range.first, -2*Lx*Ly*3*3*3*3);
   EXPECT_EQ(model.normalized_energy_range.second, 2*Lx*Ly*3*3*3*3);

}

};
};
