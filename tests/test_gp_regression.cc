// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <vector>

double test_gp_regression(libgp::GaussianProcess * gp)
{
  int input_dim = gp->get_input_dim();
  int n = libgp::Utils::randi(400)+100;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n*0.8; ++i) {
    std::vector<double> x(input_dim);
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(&x[0], y(i));
  }
  double tss = 0;
  for(int i = n*0.8+1; i < n; ++i) {
    std::vector<double> x(input_dim);
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    double f = gp->f(&x[0]);
    double error = f - y(i);
    tss += error*error;
  }
  return tss/(n*0.2-1);
}

void run_regression_test(std::string covf_str)
{
  double mss = 0.0;
  int n=20;
  for (int i=0; i<n; ++i) {
    int input_dim = libgp::Utils::randi(2) + 2;
    libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, covf_str);
    Eigen::VectorXd params(gp->covf().get_param_dim());
    params.setZero();
    params(gp->covf().get_param_dim()-1) = -2;
    gp->covf().set_loghyper(params);
    mss += test_gp_regression(gp);    
    delete gp;
  }
  ASSERT_TRUE(mss/n < 0.05);
}

TEST(GPRegressionTest, SEiso) {
  std::string covf_str("CovSum ( CovSEiso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, Matern3iso) {
  std::string covf_str("CovSum ( CovMatern3iso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, Matern5iso) {
  std::string covf_str("CovSum ( CovMatern5iso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, CovSEard) {
  std::string covf_str("CovSum ( CovSEard, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, CovRQiso) {
  std::string covf_str("CovSum ( CovRQiso, CovNoise)");
  run_regression_test(covf_str);
}

TEST(GPRegressionTest, UpdateL) {
  int input_dim = 2;
  libgp::GaussianProcess * gp = new libgp::GaussianProcess(input_dim, "CovSum ( CovSEiso, CovNoise)");
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params << 0, 0, -2;
  gp->covf().set_loghyper(params);
  size_t n = 10;
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd y = gp->covf().draw_random_sample(X);
  for(size_t i = 0; i < n; ++i) {
    std::vector<double> x(input_dim);
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp->add_pattern(&x[0], y(i));
  }
  std::vector<double> x(2, 0);
  gp->f(&x[0]);
}

TEST(GPTest, TestRegression) {
  // initialize gaussian process for 2-D input using a squared exponential covariance function
  libgp::GaussianProcess* gp = new libgp::GaussianProcess(2, "CovSEiso");
  
  // Set hyperparameters (length scale and signal variance)
  Eigen::VectorXd params(2);
  params << 0.0, 0.0;  // log-space parameters
  gp->covf().set_loghyper(params);
  
  // Check if input dimension is correct
  int input_dim = gp->get_input_dim();
  ASSERT_EQ(2, input_dim);
  
  // create inputs
  std::vector<double> x(input_dim);
  // test set target
  double y = 1.0;
  // Set inputs
  x[0] = 0.3;
  x[1] = 0.4;
  // Add training data
  gp->add_pattern(&x[0], y);
  // Add more training data
  x[0] = 0.35;
  x[1] = 0.45;
  gp->add_pattern(&x[0], y);
  
  // Test prediction
  std::vector<double> x_test(input_dim);
  x_test[0] = 0.32;
  x_test[1] = 0.42;
  double prediction = gp->f(&x_test[0]);
  ASSERT_NEAR(1.0, prediction, 0.1);
  delete gp;
}
