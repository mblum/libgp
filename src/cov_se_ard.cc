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

#include "cov_se_ard.h"
#include <cmath>

namespace libgp
{

CovSEard::CovSEard() {}

CovSEard::~CovSEard() {}

bool CovSEard::init(int n)
{
	input_dim = n;
	param_dim = n+1;
	ell.resize(input_dim);
	loghyper.resize(param_dim);
  return true;
}

double CovSEard::get(Eigen::VectorXd &x1, Eigen::VectorXd &x2)
{  
	double z = (x1-x2).cwiseQuotient(ell).squaredNorm();
	return sf2*exp(-0.5*z);
}

void CovSEard::grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad)
{
  Eigen::VectorXd z = (x1-x2).cwiseQuotient(ell).array().square();  
  double k = sf2*exp(-0.5*z.sum());
  grad.head(input_dim) = z * k;
  grad(input_dim) = 2.0 * k;
}

bool CovSEard::set_loghyper(Eigen::VectorXd &p)
{
  if (!CovarianceFunction::set_loghyper(p)) return false;
	for(size_t i = 0; i < input_dim; ++i) ell(i) = exp(loghyper(i));
	sf2 = exp(2*loghyper(input_dim));
	return true;
}

std::string CovSEard::to_string()
{
	return "CovSEard";
}
}

