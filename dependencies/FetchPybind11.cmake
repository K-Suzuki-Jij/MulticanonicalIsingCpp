include(FetchContent)

message(STATUS "Downloading pybind11...")
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)
message(STATUS "pybind11 download complete.")