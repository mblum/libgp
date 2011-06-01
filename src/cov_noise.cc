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

#include "cov_noise.h"
#include <cmath>

namespace libgp
{

CovNoise::CovNoise() {}

CovNoise::~CovNoise() {}

bool CovNoise::init(int n)
{
	input_dim = n;
	param_dim = 1;
	loghyper.resize(param_dim);
  return true;
}

double CovNoise::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
{
	if (&x1 == &x2) return s2;
	else return 0.0;
}

void CovNoise::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
{
	if (&x1 == &x2) grad(0) = 2*s2;
	else grad(0) = 0.0;
}

void CovNoise::set_loghyper(const Eigen::VectorXd &p)
{
  CovarianceFunction::set_loghyper(p);
	s2 = exp(2*loghyper(0));
}

std::string CovNoise::to_string()
{
	return "CovNoise";
}

}
