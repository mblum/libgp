// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "rprop.h"
#include "gp_utils.h"

namespace libgp {

void RProp::init(double Delta0, double Deltamin, double Deltamax, double etaminus, double etaplus) 
{
  this->Delta0   = Delta0;
  this->Deltamin = Deltamin;
  this->Deltamax = Deltamax;
  this->etaminus = etaminus;
  this->etaplus  = etaplus;
}

void RProp::maximize(GaussianProcess * gp, size_t n, bool verbose)
{
  int param_dim = gp->covf().get_param_dim();
  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
  Eigen::VectorXd params = gp->covf().get_loghyper();

  for (size_t i=0; i<n; ++i) {
    if (verbose) std::cout << -gp->log_likelihood() << std::endl;
    Eigen::VectorXd grad = -gp->log_likelihood_gradient();
    grad_old = grad_old.cwiseProduct(grad);
    for (int j=0; j<grad_old.size(); ++j) {
      if (grad_old(j) > 0) {
        Delta(j) = std::min(Delta(j)*etaplus, Deltamax);        
      } else if (grad_old(j) < 0) {
        Delta(j) = std::max(Delta(j)*etaminus, Deltamin);
        grad(j) = 0;
      } 
      params(j) += -Utils::sign(grad(j)) * Delta(j);
    }
    grad_old = grad;
    gp->covf().set_loghyper(params);
  }
  std::cout << params << std::endl;
}

}
