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

#include "cov_sum.h"
#include "cmath"

namespace libgp
{

CovSum::CovSum()
{
}

CovSum::~CovSum()
{
	delete first;
	delete second;
}

void CovSum::init(int n, CovarianceFunction * first, CovarianceFunction * second)
{
	this->input_dim = n;
	this->first = first;
	this->second = second;
	param_dim_first = first->get_param_dim();
	param_dim_second = second->get_param_dim();
	param_dim = param_dim_first + param_dim_second;
	loghyper.resize(param_dim);
}

double CovSum::get(Eigen::VectorXd &x1, Eigen::VectorXd &x2)
{
	return first->get(x1, x2) + second->get(x1, x2);
}

void CovSum::grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad)
{
  Eigen::VectorXd grad_first = grad.segment(0, param_dim_first);
  Eigen::VectorXd grad_second = grad.segment(param_dim_first, param_dim_second);
	first->grad(x1, x2, grad_first);
	second->grad(x1, x2, grad_second);
}

bool CovSum::set_loghyper(Eigen::VectorXd &p)
{
	if (CovarianceFunction::set_loghyper(p)) {
    Eigen::VectorXd param_first = p.segment(0, param_dim_first);
    Eigen::VectorXd param_second = p.segment(param_dim_first, param_dim_second);
		first->set_loghyper(param_first);
		second->set_loghyper(param_second);
		return true;
	}
	return false;
}

std::string CovSum::to_string()
{
	return "Sum("+first->to_string()+", "+second->to_string()+")";
}

}