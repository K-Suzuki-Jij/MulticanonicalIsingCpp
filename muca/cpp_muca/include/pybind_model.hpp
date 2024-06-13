#pragma once

#include "../../../include/all.hpp"

namespace cpp_muca {
namespace wrapper {

namespace py = pybind11;
//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

void PyBindPBodyTwoDimIsing(py::module &m) {
   auto py_class = py::class_<PBodyTwoDimIsing>(m, "PBodyTwoDimIsing", py::module_local());

   // Constructors
   py_class.def(py::init<const double, const int, const int, const int, const double, const double>(),
                "J"_a, "p"_a, "Lx"_a, "Ly"_a, "spin"_a, "spin_scale_factor"_a);

   //Public Member Variables
   py_class.def_readonly("J", &PBodyTwoDimIsing::J);
   py_class.def_readonly("p", &PBodyTwoDimIsing::p);
   py_class.def_readonly("Lx", &PBodyTwoDimIsing::Lx);
   py_class.def_readonly("Ly", &PBodyTwoDimIsing::Ly);
   py_class.def_readonly("spin", &PBodyTwoDimIsing::spin);
   py_class.def_readonly("spin_scale_factor", &PBodyTwoDimIsing::spin_scale_factor);
   py_class.def_readonly("normalized_energy_range", &PBodyTwoDimIsing::normalized_energy_range);
   
}

} // namespace wrapper
} // namespace cpp_muca
