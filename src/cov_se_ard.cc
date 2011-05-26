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

CovSEard::~CovSEard()
{
	delete [] ell;
}

void CovSEard::init(int n)
{
	input_dim = n;
	param_dim = n+1;
	ell = new double[param_dim-1];
	loghyper.resize(param_dim);
}

double CovSEard::get(Eigen::VectorXd &x1, Eigen::VectorXd &x2)
{
	double z = 0.0;
	for(size_t i = 0; i < input_dim; ++i) {
		z += pow((x1(i)-x2(i))/ell[i], 2);
	}
	return sf2*exp(-0.5*z);
}

void CovSEard::grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad)
{
	double z = 0.0, k;
	for(size_t i = 0; i < input_dim; ++i) {
		z += pow((x1(i)-x2(i))/ell[i], 2);
		grad(i) = z;
	}
  grad(input_dim) = 2.0;
	k = sf2*exp(-0.5*z);
  grad = grad/k;
}

bool CovSEard::set_loghyper(Eigen::VectorXd &p)
{
	bool a = CovarianceFunction::set_loghyper(p);
	for(size_t i = 0; i < param_dim-1; ++i) {
		ell[i] = exp(loghyper(i));
	}
	sf2 = exp(2*loghyper(param_dim-1));
	return a;
}

std::string CovSEard::to_string()
{
	return "SEard";
}
}

