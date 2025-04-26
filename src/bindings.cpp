#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "gp.h"
#include "cov_factory.h"

namespace py = pybind11;

PYBIND11_MODULE(libgp_cpp, m) {
    m.doc() = "Python bindings for libgp - Gaussian Process Regression Library";

    py::class_<libgp::GaussianProcess>(m, "GaussianProcess")
        .def(py::init<>())
        .def(py::init<size_t, const std::string&>())
        .def(py::init<const char*>())
        .def("add_pattern", [](libgp::GaussianProcess& self, py::array_t<double> x, double y) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 1) 
                throw std::runtime_error("Input array must be 1-dimensional");
            double* ptr = static_cast<double*>(buf.ptr);
            self.add_pattern(ptr, y);
        })
        .def("predict", [](libgp::GaussianProcess& self, py::array_t<double> x) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 1)
                throw std::runtime_error("Input array must be 1-dimensional");
            double* ptr = static_cast<double*>(buf.ptr);
            return self.f(ptr);
        })
        .def("get_variance", [](libgp::GaussianProcess& self, py::array_t<double> x) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 1)
                throw std::runtime_error("Input array must be 1-dimensional");
            double* ptr = static_cast<double*>(buf.ptr);
            return self.var(ptr);
        })
        .def("set_y", &libgp::GaussianProcess::set_y)
        .def("get_sampleset_size", &libgp::GaussianProcess::get_sampleset_size)
        .def("clear_sampleset", &libgp::GaussianProcess::clear_sampleset)
        .def("get_log_likelihood", &libgp::GaussianProcess::log_likelihood)
        .def("get_log_likelihood_gradient", &libgp::GaussianProcess::log_likelihood_gradient)
        .def("get_input_dim", &libgp::GaussianProcess::get_input_dim)
        .def("set_loghyper", [](libgp::GaussianProcess& self, py::array_t<double> params) {
            py::buffer_info buf = params.request();
            if (buf.ndim != 1)
                throw std::runtime_error("Parameter array must be 1-dimensional");
            Eigen::Map<const Eigen::VectorXd> eigen_params(static_cast<double*>(buf.ptr), buf.shape[0]);
            self.covf().set_loghyper(eigen_params);
        })
        .def("get_loghyper", [](libgp::GaussianProcess& self) {
            return py::array_t<double>(
                {static_cast<py::ssize_t>(self.covf().get_param_dim())},
                {sizeof(double)},
                self.covf().get_loghyper().data()
            );
        })
        .def("get_param_dim", [](libgp::GaussianProcess& self) {
            return self.covf().get_param_dim();
        });

    py::class_<libgp::CovFactory>(m, "CovFactory")
        .def(py::init<>())
        .def("create", &libgp::CovFactory::create)
        .def("list", &libgp::CovFactory::list);
}