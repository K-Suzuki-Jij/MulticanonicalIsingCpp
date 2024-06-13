cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(muca DESCRIPTION "Multicanonical Simulation Library")

set(MUCA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_VERBOSE_MAKEFILE TRUE)


# pybind11
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
if(PYBIND11_FOUND)
  message(STATUS "Skip Download pybind11")
else()
  include(dependencies/FetchPybind11.cmake)
endif()

# OpenMP
if(APPLE)
  message(STATUS "Enable OpenMP forcely from Homebrew on Apple")
  execute_process(COMMAND brew --prefix libomp COMMAND tr -d '\n' OUTPUT_VARIABLE OMP_DIR)
  include_directories("${OMP_DIR}/include")
  link_directories("${OMP_DIR}/lib")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
else()
  find_package(OpenMP REQUIRED)
endif()

add_subdirectory(include)
add_subdirectory(muca)

