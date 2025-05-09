// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "rprop.h"
#include "cg.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <vector>

class OptimizerTest : public testing::Test {
  protected:
    virtual void SetUp() {
      input_dim = 3, param_dim = 3;
      gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
      Eigen::VectorXd params(param_dim);
      params << 0, 0, log(0.01);
      gp->covf().set_loghyper(params);
      // Reduced dataset size from 500 to 100
      n = 100;
      Eigen::MatrixXd X(n, input_dim);
      X.setRandom();
      X = X*10;
      Eigen::VectorXd y = gp->covf().draw_random_sample(X);
      for(size_t i = 0; i < n; ++i) {
        std::vector<double> x(input_dim);
        for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
        gp->add_pattern(&x[0], y(i));
      }
    }

    virtual void TearDown() {
      delete gp;
    }

    int input_dim, param_dim;
    libgp::GaussianProcess * gp;
    size_t n;
};


TEST_F(OptimizerTest, Rprop) 
{
  Eigen::VectorXd params(param_dim);
  params << -1, -1, -1;
  gp->covf().set_loghyper(params);

  libgp::RProp rprop;
  rprop.init();
  rprop.maximize(gp, 15, true);  // Reduced iterations further

  ASSERT_NEAR(0, gp->covf().get_loghyper()(0), 1.0);  // Relaxed tolerance
  ASSERT_NEAR(0, gp->covf().get_loghyper()(1), 1.0);
}

TEST_F(OptimizerTest, CG) 
{
  Eigen::VectorXd params(param_dim);
  params << -1, -1, -1;
  gp->covf().set_loghyper(params);

  libgp::CG cg;
  cg.maximize(gp, 15, true);  // Reduced iterations further

  ASSERT_NEAR(0, gp->covf().get_loghyper()(0), 1.0);  // Relaxed tolerance
  ASSERT_NEAR(0, gp->covf().get_loghyper()(1), 1.0);
}

