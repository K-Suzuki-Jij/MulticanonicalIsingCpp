#include "gtest/gtest.h"

#include "test_algorithm/test_multicanonical.hpp"
#include "test_algorithm/test_parameters.hpp"
#include "test_algorithm/test_util.hpp"
#include "test_algorithm/test_wang_landau.hpp"
#include "test_model/test_p_body_ising.hpp"

int main(std::int32_t argc, char **argv) {
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
