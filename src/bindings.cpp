#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "gp.h"
#include "cov_factory.h"
#include "rprop.h"
#include "cg.h"

namespace py = pybind11;

PYBIND11_MODULE(libgp_cpp, m) {
    m.doc() = "Python bindings for libgp - Gaussian Process Regression Library";

    py::class_<libgp::GaussianProcess>(m, "GaussianProcess")
        .def(py::init<size_t, const std::string&>())
        .def("add_pattern", [](libgp::GaussianProcess& self, py::array_t<double> x, double y) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 1) 
                throw std::runtime_error("Input array must be 1-dimensional");
            double* ptr = static_cast<double*>(buf.ptr);
            self.add_pattern(ptr, y);
        })
        .def("add_patterns", [](libgp::GaussianProcess& self, py::array_t<double> x, py::array_t<double> y) {
            py::buffer_info buf_x = x.request();
            py::buffer_info buf_y = y.request();
            if (buf_x.ndim != 2 || buf_y.ndim != 1)
                throw std::runtime_error("Input matrix must be 2-dimensional and output vector must be 1-dimensional");
            if (buf_x.shape[0] != buf_y.shape[0])
                throw std::runtime_error("Number of input patterns must match number of target values");
            double* ptr_x = static_cast<double*>(buf_x.ptr);
            double* ptr_y = static_cast<double*>(buf_y.ptr);
            Eigen::Map<const Eigen::MatrixXd> eigen_x(ptr_x, buf_x.shape[0], buf_x.shape[1]);
            Eigen::Map<const Eigen::VectorXd> eigen_y(ptr_y, buf_y.shape[0]);
            self.add_patterns(eigen_x, eigen_y);
        })
        .def("predict", [](libgp::GaussianProcess& self, py::array_t<double> x, bool compute_variance) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Input array must be 2-dimensional");
            double* ptr = static_cast<double*>(buf.ptr);
            Eigen::Map<const Eigen::MatrixXd> eigen_x(ptr, buf.shape[0], buf.shape[1]);
            return self.predict(eigen_x, compute_variance);
        })
        .def("set_y", &libgp::GaussianProcess::set_y)
        .def("get_sampleset_size", &libgp::GaussianProcess::get_sampleset_size)
        .def("clear_sampleset", &libgp::GaussianProcess::clear_sampleset)
        .def("get_sampleset", &libgp::GaussianProcess::get_sampleset)
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
        })
        .def("get_covariance_function", [](libgp::GaussianProcess& self) {
            return self.covf().to_string();
        });

    py::class_<libgp::CovFactory>(m, "CovFactory")
        .def(py::init<>())
        .def("create", &libgp::CovFactory::create)
        .def("list", &libgp::CovFactory::list);

    py::class_<libgp::RProp>(m, "RProp")
        .def(py::init<>())
        .def("init", &libgp::RProp::init)
        .def("maximize", &libgp::RProp::maximize);

    py::class_<libgp::CG>(m, "CG")
        .def(py::init<>())
        .def("maximize", &libgp::CG::maximize);
}