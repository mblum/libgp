// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "rprop.h"
#include "gp_utils.h"

namespace libgp {

void RProp::init(double eps_stop, double Delta0, double Deltamin, double Deltamax, double etaminus, double etaplus, double min_stepsize_factor)
{
  this->Delta0   = Delta0;
  this->Deltamin = Deltamin;
  this->Deltamax = Deltamax;
  this->etaminus = etaminus;
  this->etaplus  = etaplus;
  this->eps_stop = eps_stop;
  this->min_stepsize_factor = min_stepsize_factor;
}

void RProp::maximize(GaussianProcess * gp, size_t n, bool verbose, bool print_params)
{
  int param_dim = gp->covf().get_param_dim();
  Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
  Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
  Eigen::VectorXd params = gp->covf().get_loghyper();
  Eigen::VectorXd best_params = params;
  double best = log(0);

  float stepsize_factor = 1.0;

  for (size_t i=0; i<n; ++i) {
    Eigen::VectorXd grad = -stepsize_factor * gp->log_likelihood_gradient();
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
    if (grad_old.norm() < eps_stop) break;
    gp->covf().set_loghyper(params);
    double lik = gp->log_likelihood();
    if (verbose) std::cout << i << " " << -lik << std::endl;
    if (print_params) {
      std::cout << "[" << params[0];
      for (int ii = 1; ii < params.size(); ++ii) {
        std::cout << ", " << params[ii];
      }
      std::cout << "]" << std::endl;
    }
    if (lik > best) {
      best = lik;
      best_params = params;
    }
    else {
      stepsize_factor /= 2;
      if (verbose) std::cout << "no improvement in step " << i
                             << ", reduced stepsize_factor to " << stepsize_factor << std::endl;
      if (stepsize_factor < min_stepsize_factor) {
        if (verbose) std::cout << "stepsize_factor below threshold, finishing" << std::endl;
        break;
      }
    }
  }
  gp->covf().set_loghyper(best_params);
}

} // namespace libgp
