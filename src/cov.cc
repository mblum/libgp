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

}