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

#include "cov_se_iso.h"
#include <cmath>

namespace libgp
{

CovSEiso::CovSEiso() {}

CovSEiso::~CovSEiso() {}

void CovSEiso::init(int n)
{
	input_dim = n;
	param_dim = 2;
	loghyper.resize(param_dim);
}

double CovSEiso::get(Eigen::VectorXd &x1, Eigen::VectorXd &x2)
{
	double z = 0.0;
	for(size_t i = 0; i < input_dim; ++i) {
		z += pow((x1(i)-x2(i))/ell, 2);
	}
	return sf2*exp(-0.5*z);
}

void CovSEiso::grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad)
{
	double z = 0.0, k;
	for(size_t i = 0; i < input_dim; ++i) {
		z += pow((x1(i)-x2(i))/ell, 2);
	}
	k = sf2*exp(-0.5*z);
	grad(0) = k*z;
	grad(0) = 2*k;
}

bool CovSEiso::set_loghyper(Eigen::VectorXd &p)
{
	bool a = CovarianceFunction::set_loghyper(p);
	ell = exp(loghyper(0));
	sf2 = exp(2*loghyper(1));
	return a;
}

std::string CovSEiso::to_string()
{
	return "SEiso";
}

}