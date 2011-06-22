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

#include "cov_se_iso.h"
#include <cmath>

namespace libgp
{
  
  CovSEiso::CovSEiso() {}
  
  CovSEiso::~CovSEiso() {}
  
  bool CovSEiso::init(int n)
  {
    input_dim = n;
    param_dim = 2;
    loghyper.resize(param_dim);
    loghyper.setZero();
    return true;
  }
  
  double CovSEiso::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    return sf2*exp(-0.5*z);
  }
  
  void CovSEiso::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    double z = ((x1-x2)/ell).squaredNorm();
    double k = sf2*exp(-0.5*z);
    grad << k*z, 2*k;
  }
  
  void CovSEiso::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    ell = exp(loghyper(0));
    sf2 = exp(2*loghyper(1));
  }
  
  std::string CovSEiso::to_string()
  {
    return "CovSEiso";
  }
  
}