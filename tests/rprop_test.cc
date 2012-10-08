// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "rprop.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

TEST(RPropTest, Test1) 
{
  int input_dim = 3, param_dim = 5;
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEard, CovNoise)");
  Eigen::VectorXd params(param_dim);
  params << 1, 2, 3, 0, -2;
  gp->covf().set_loghyper(params);
  int n = 400;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
  }

  params << 0, 0, 0, 0, 0;

  gp->covf().set_loghyper(params);
  
  libgp::RProp rprop;
  rprop.init();
  rprop.maximize(gp);

  std::cout << gp->covf().get_loghyper() << std::endl;

}


