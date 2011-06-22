/**************************************************************
 libgp - Gaussian process library for Machine Learning
 Copyright (C) 2011 Universität Freiburg
 Author: Manuel Blum
 
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 ***************************************************************/

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
  
  bool CovSum::init(int n, CovarianceFunction * first, CovarianceFunction * second)
  {
    this->input_dim = n;
    this->first = first;
    this->second = second;
    param_dim_first = first->get_param_dim();
    param_dim_second = second->get_param_dim();
    param_dim = param_dim_first + param_dim_second;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovSum::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    return first->get(x1, x2) + second->get(x1, x2);
  }
  
  void CovSum::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd grad_first(param_dim_first);
    Eigen::VectorXd grad_second(param_dim_second);
    first->grad(x1, x2, grad_first);
    second->grad(x1, x2, grad_second);
    grad.head(param_dim_first) = grad_first;
    grad.tail(param_dim_second) = grad_second;
  }
  
  void CovSum::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    first->set_loghyper(p.head(param_dim_first));
    second->set_loghyper(p.tail(param_dim_second));
  }
  
  std::string CovSum::to_string()
  {
    return "CovSum("+first->to_string()+", "+second->to_string()+")";
  }
  
  double CovSum::get_threshold()
  {
    return std::max(first->get_threshold(), second->get_threshold());
  }
  
  void CovSum::set_threshold(double threshold) 
  {
    first->set_threshold(threshold);
    second->set_threshold(threshold);
  }
  
}