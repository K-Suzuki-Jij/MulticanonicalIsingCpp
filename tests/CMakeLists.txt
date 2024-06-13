add_executable(cxx_muca_test
    cpp_tests/test_main.cpp
)

if(ENABLE_COVERAGE AND (NOT MSVC))
  add_coverage(cxx_muca_test)
endif()

target_include_directories(cxx_muca_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(cxx_muca_test PRIVATE
    BUILD_TESTING=OFF
)

target_link_libraries(cxx_muca_test PRIVATE
    cxx_muca
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,gtest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,gtest_main>
    $<IF:$<TARGET_EXISTS:GTest::gmock>,GTest::gmock,gmock>
    $<IF:$<TARGET_EXISTS:GTest::gmock_main>,GTest::gmock_main,gmock_main>
)

gtest_discover_tests(cxx_muca_test)
