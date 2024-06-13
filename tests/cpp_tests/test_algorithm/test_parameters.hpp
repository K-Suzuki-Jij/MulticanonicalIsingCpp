#pragma once

#include "../../../include/algorithm/parameters.hpp"
#include "gtest/gtest.h"

namespace cpp_muca {
namespace test {

TEST(AlgorithmParameters, WangLandauParameters) {
   
   const auto p = WangLandauParameters(1e-08, 1000, 3, 999, 100, 0.65, 0.2, 0.1);
   EXPECT_EQ(p.modification_criterion    , 1e-08);
   EXPECT_EQ(p.convergence_check_interval, 1000 );
   EXPECT_EQ(p.num_divided_energy_range  , 3    );
   EXPECT_EQ(p.seed                      , 999  );
   EXPECT_EQ(p.max_sweeps                , 100  );
   EXPECT_EQ(p.flatness_criterion        , 0.65 );
   EXPECT_EQ(p.reduce_rate               , 0.2  );
   EXPECT_EQ(p.overlap_rate              , 0.1  );
   
   EXPECT_THROW(WangLandauParameters(0.0  , 1000, 3, 999, 100, 0.65, 0.2, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, -1  , 3, 999, 100, 0.65, 0.2, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, 1000, 0, 999, 100, 0.65, 0.2, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, 1000, 3, 999, -1 , 0.65, 0.2, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, 1000, 3, 999, 100, 1.0 , 0.2, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, 1000, 3, 999, 100, 0.65, 1.0, 0.1), std::invalid_argument);
   EXPECT_THROW(WangLandauParameters(1e-08, 1000, 3, 999, 100, 0.65, 0.2, 1.1), std::invalid_argument);
}


TEST(AlgorithmParameters, MulticanonicalParameters) {
   
   const auto p = MulticanonicalParameters(100, 999, 3, 0.1);
   EXPECT_EQ(p.num_sweeps              , 100);
   EXPECT_EQ(p.seed                    , 999);
   EXPECT_EQ(p.num_divided_energy_range, 3  );
   EXPECT_EQ(p.overlap_rate            , 0.1);

   EXPECT_THROW(MulticanonicalParameters(0  , 999, 3 , 0.1 ), std::invalid_argument);
   EXPECT_THROW(MulticanonicalParameters(100, 999, -1, 0.1 ), std::invalid_argument);
   EXPECT_THROW(MulticanonicalParameters(100, 999, 3 , -0.1), std::invalid_argument);
}

}
}
