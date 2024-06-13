message(CHECK_START "cxx_muca")

add_library(cxx_muca INTERFACE)

target_include_directories(cxx_muca INTERFACE 
  $<BUILD_INTERFACE:${MUCA_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(cxx_muca INTERFACE 
    $<$<TARGET_EXISTS:OpenMP::OpenMP_CXX>:OpenMP::OpenMP_CXX>
    $<$<TARGET_EXISTS:BLAS::BLAS>:BLAS::BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:LAPACK::LAPACK>
)

target_compile_definitions(cxx_muca INTERFACE 
    BUILD_TESTING=OFF
    TEST_LIB=OFF
    $<$<TARGET_EXISTS:BLAS::BLAS>:EIGEN_USE_BLAS>
    $<$<TARGET_EXISTS:LAPACK::LAPACK>:EIGEN_USE_LAPACKE>
    $<$<CXX_COMPILER_ID:MSVC>:EIGEN_Fortran_COMPILER_WORKS=OFF>   
)

message(CHECK_PASS "Finished")
