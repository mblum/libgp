//
// libgp - Gaussian Process library for Machine Learning
// Copyright (C) 2010 Universität Freiburg
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

#include "cov.h"
#include "gp_utils.h"

namespace libgp
{

size_t CovarianceFunction::get_param_dim()
{
	return param_dim;
}

size_t CovarianceFunction::get_input_dim()
{
	return input_dim;
}

Eigen::VectorXd CovarianceFunction::get_loghyper()
{
	return loghyper;
}

bool CovarianceFunction::set_loghyper(Eigen::VectorXd &p)
{
	if (p.size() != loghyper.size()) {
		std::cerr << "error: parameter vector must be of length " << param_dim << std::endl;
		return false;
	} else {
		loghyper = p;
		return true;
	}
}

void CovarianceFunction::draw_random_sample(Eigen::MatrixXd &X, Eigen::VectorXd &y)
{
  assert (X.cols() == input_dim);  
  int n = X.rows();
	Eigen::MatrixXd K(n, n);
	Eigen::LLT<Eigen::MatrixXd> solver;
	y.resize(n);
	// compute kernel matrix (lower triangle)
	for(size_t i = 0; i < n; ++i) {
		for(size_t j = i; j < n; ++j) {
      Eigen::VectorXd a = X.row(j);
      Eigen::VectorXd b = X.row(i);
      assert (a.size() == input_dim);
      assert (b.size() == input_dim);
      K(j, i) = get(a, b);
		}
		y(i) = randn();
	}
	// perform cholesky factorization
  solver = K.llt();
  y = solver.matrixL() * y;
}

}