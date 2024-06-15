#pragma once

#include "../../../include/all.hpp"

namespace cpp_muca {
namespace wrapper {

namespace py = pybind11;
//The following does not bring in anything else from the pybind11 namespace except for literals.
using namespace pybind11::literals;

void PyBindWangLandauParameters(py::module &m) {
   auto py_class = py::class_<WangLandauParameters>(m, "WangLandauParameters", py::module_local());

   // Constructors
   py_class.def(py::init<const double, const int, const int, const std::size_t, const std::int64_t, const double, const double, const double>(),
                "modification_criterion"_a, 
                "convergence_check_interval"_a, 
                "num_divided_energy_range"_a, 
                "seed"_a,
                "max_sweeps"_a=std::numeric_limits<std::int64_t>::max(), 
                "flatness_criterion"_a=0.8, 
                "reduce_rate"_a=0.5, 
                "overlap_rate"_a=0.2
                );

   //Public Member Variables
   py_class.def_readonly("modification_criterion", &WangLandauParameters::modification_criterion);
   py_class.def_readonly("convergence_check_interval", &WangLandauParameters::convergence_check_interval);
   py_class.def_readonly("num_divided_energy_range", &WangLandauParameters::num_divided_energy_range);
   py_class.def_readonly("seed", &WangLandauParameters::seed);
   py_class.def_readonly("max_sweeps", &WangLandauParameters::max_sweeps);
   py_class.def_readonly("flatness_criterion", &WangLandauParameters::flatness_criterion);
   py_class.def_readonly("reduce_rate", &WangLandauParameters::reduce_rate);
   py_class.def_readonly("overlap_rate", &WangLandauParameters::overlap_rate);
   
}

void PyBindBaseWangLandauResults(py::module &m) {
   auto py_class = py::class_<BaseWangLandauResults>(m, "BaseWangLandauResults", py::module_local());

   // Constructors
   py_class.def(py::init<const std::unordered_map<int, double>&, const OrderParameters&, const std::int64_t, const double>(),
                "entropy_dict"_a,
                "order_parameters"_a, 
                "total_sweeps"_a, 
                "final_modification_factor"_a
                );

   //Public Member Variables
   py_class.def_readonly("entropy_dict", &BaseWangLandauResults::entropy_dict);
   py_class.def_readonly("order_parameters", &BaseWangLandauResults::order_parameters);
   py_class.def_readonly("total_sweeps", &BaseWangLandauResults::total_sweeps);
   py_class.def_readonly("final_modification_factor", &BaseWangLandauResults::final_modification_factor);
}

void PyBindOrderParameters(py::module &m) {
   auto py_class = py::class_<OrderParameters>(m, "OrderParameters", py::module_local());

   // Constructors
   py_class.def(py::init<const std::unordered_map<int, double>&, const std::unordered_map<int, double>&, const std::unordered_map<int, double>&, const std::unordered_map<int, double>&, const std::unordered_map<int, std::int64_t>&>(),
                "mag_2"_a,
                "mag_4"_a,
                "abs_f2"_a,
                "abs_f4"_a,
                "normalized_energy_count"_a
                );

   //Public Member Variables
   py_class.def_readonly("mag_2", &OrderParameters::mag_2);
   py_class.def_readonly("mag_4", &OrderParameters::mag_4);
   py_class.def_readonly("abs_f2", &OrderParameters::abs_f2);
   py_class.def_readonly("abs_f4", &OrderParameters::abs_f4);
   py_class.def_readonly("normalized_energy_count", &OrderParameters::normalized_energy_count);

}

void PyBindWangLandau(py::module &m) {
   m.def("run_wang_landau", &WangLandau, "model"_a, "parameters"_a, "num_threads"_a, "calculate_order_parameters"_a);
}

void PyBindMulticanonicalParameters(py::module &m) {
   auto py_class = py::class_<MulticanonicalParameters>(m, "MulticanonicalParameters", py::module_local());

   // Constructors
   py_class.def(py::init<const std::int64_t, const std::size_t, const int, const double>(),
                "num_sweeps"_a,
                "seed"_a, 
                "num_divided_energy_range"_a, 
                "overlap_rate"_a
                );

   //Public Member Variables
   py_class.def_readonly("num_sweeps", &MulticanonicalParameters::num_sweeps);
   py_class.def_readonly("seed", &MulticanonicalParameters::seed);
   py_class.def_readonly("num_divided_energy_range", &MulticanonicalParameters::num_divided_energy_range);
   py_class.def_readonly("overlap_rate", &MulticanonicalParameters::overlap_rate);

}

void PyBindBaseMulticanonicalResults(py::module &m) {
   auto py_class = py::class_<BaseMulticanonicalResults>(m, "BaseMulticanonicalResults", py::module_local());

   // Constructors
   py_class.def(py::init<const std::unordered_map<int, std::int64_t>&, const OrderParameters&>(),
                "histogram_dict"_a,
                "order_parameters"_a
                );

   //Public Member Variables
   py_class.def_readonly("histogram_dict", &BaseMulticanonicalResults::histogram_dict);
   py_class.def_readonly("order_parameters", &BaseMulticanonicalResults::order_parameters);

}

void PyBindMulticanonical(py::module &m) {
   m.def("run_multicanonical", &Multicanonical, "model"_a, "parameters"_a, "entropy_dict"_a, "num_threads"_a, "calculate_order_parameters"_a);
}


} // namespace wrapper
} // namespace cpp_muca
