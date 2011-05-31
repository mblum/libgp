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

#define F(X, Y) (X * exp(-X*X - Y*Y))

#include "gp.h"
#include "cov_factory.h"

#include "cmath"
#include <iostream>
#include <gtest/gtest.h>

void test_gp_regression(int n, std::string covf_str, int input_dim, double params[])
{
  libgp::CovFactory factory;
  libgp::CovarianceFunction * covf = factory.create(input_dim, covf_str);
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd p(covf->get_param_dim());
  for(size_t i = 0; i < covf->get_param_dim(); ++i) p(i) = params[i];
  Eigen::VectorXd y(n);
  covf->set_loghyper(p);
  covf->draw_random_sample(X, y);
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, covf_str);    
  gp->set_params(params);
  for(size_t i = 0; i < n*0.8; ++i) {
    double x[input_dim];
    for(size_t j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(x, y(i));
  }
	double tss = 0;
  for(size_t i = n*0.8+1; i < n; ++i) {
    double x[input_dim];
    for(size_t j = 0; j < input_dim; ++j) x[j] = X(i,j);
    double f = gp->predict(x);
    double error = f - y(i);
    tss += error*error;
  }
	delete gp;
  ASSERT_LE(tss/n, 10e-4);
}

TEST(GPRegressionTest, SEiso) {
  std::string covf_str("CovSum ( CovSEiso, CovNoise)");
  double params[3] = {0, 0, -2.3};
  test_gp_regression(200, covf_str, 2, params);
  test_gp_regression(500, covf_str, 2, params);
  test_gp_regression(800, covf_str, 3, params);
  test_gp_regression(1000, covf_str, 3, params);
}

TEST(GPRegressionTest, Matern3iso) {
  std::string covf_str("CovSum ( CovMatern3iso, CovNoise)");
  double params[3] = {0, 0, -2.3};
  test_gp_regression(200, covf_str, 2, params);
  test_gp_regression(500, covf_str, 2, params);
  test_gp_regression(800, covf_str, 3, params);
  test_gp_regression(1000, covf_str, 3, params);
}
