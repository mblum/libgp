/**************************************************************
 libgp - Gaussian Process library for Machine Learning
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

#include "cov_matern3_iso.h"
#include <cmath>

namespace libgp
{
  
  CovMatern3iso::CovMatern3iso() {}
  
  CovMatern3iso::~CovMatern3iso() {}
  
  bool CovMatern3iso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    sqrt3 = sqrt(3);
    return true;
  }
  
  double CovMatern3iso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)*sqrt3/ell).norm();
    return sf2*exp(-z)*(1+z);
  }
  
  void CovMatern3iso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)*sqrt3/ell).norm();
    double k = sf2*exp(-z);
    grad << k*z*z, 2*k*(1+z);
  }
  
  void CovMatern3iso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovMatern3iso::to_string()
  {
    return "CovMatern3iso";
  }
  
}