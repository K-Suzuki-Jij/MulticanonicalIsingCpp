#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "include/pybind_model.hpp"
#include "include/pybind_algorithm.hpp"


PYBIND11_MODULE(cpp_muca, m) {
   namespace py = pybind11;
   
   py::module_ m_model = m.def_submodule("cpp_model");
   cpp_muca::wrapper::PyBindPBodyTwoDimIsing(m_model);

   py::module_ m_algorithm = m.def_submodule("cpp_algorithm");
   cpp_muca::wrapper::PyBindUpdateMethod(m_algorithm);
   cpp_muca::wrapper::PyBindWangLandauParameters(m_algorithm);
   cpp_muca::wrapper::PyBindBaseWangLandauResults(m_algorithm);
   cpp_muca::wrapper::PyBindWangLandau(m_algorithm);
   cpp_muca::wrapper::PyBindOrderParameters(m_algorithm);
   cpp_muca::wrapper::PyBindMulticanonicalParameters(m_algorithm);
   cpp_muca::wrapper::PyBindBaseMulticanonicalResults(m_algorithm);
   cpp_muca::wrapper::PyBindMulticanonical(m_algorithm);


};
