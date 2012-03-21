// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

const int input_dim = 5;
const double tol = 10e-12;

libgp::CovFactory factory;
libgp::CovarianceFunction * covf;

void test_covf(int param_dim, 
               Eigen::VectorXd &p, 
               Eigen::Matrix2d &K,
               Eigen::VectorXd &g1, 
               Eigen::VectorXd &g2)
{
  Eigen::VectorXd a(input_dim);
  Eigen::VectorXd b(input_dim);
  Eigen::VectorXd g(param_dim);
  a << 0.9143,-0.0292,0.6006,-0.7162,-0.1565;
  b << 0.8315,0.5844,0.9190,0.3115,-0.9286;
  ASSERT_EQ(input_dim, covf->get_input_dim());
  ASSERT_EQ(param_dim, covf->get_param_dim());
  covf->set_loghyper(p);
  ASSERT_NEAR(K(0,1), covf->get(a, b), tol);
  ASSERT_NEAR(K(1,0), covf->get(b, a), tol);
  ASSERT_NEAR(K(0,0), covf->get(a, a), tol);
  ASSERT_NEAR(K(1,1), covf->get(b, b), tol);
  covf->grad(a, a, g);
  for(int i = 0; i < param_dim; ++i) ASSERT_NEAR(g(i), g1(i), tol);
  covf->grad(a, b, g);
  for(int i = 0; i < param_dim; ++i) ASSERT_NEAR(g(i), g2(i), tol);
  covf->grad(b, b, g);
  for(int i = 0; i < param_dim; ++i) ASSERT_NEAR(g(i), g1(i), tol);
  covf->grad(b, a, g);
  for(int i = 0; i < param_dim; ++i) ASSERT_NEAR(g(i), g2(i), tol);
}

TEST(CovTest, Noise) {
  const int param_dim = 1;
  Eigen::VectorXd p(param_dim);
  p << -2.0;
  Eigen::Matrix2d K;
  K << 0.018315638889, 0.0, 0.0, 0.018315638889;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.036631277777;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.0;
	covf = factory.create(input_dim, "CovNoise");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, Linearone) {
  const int param_dim = 1;
  Eigen::VectorXd p(param_dim);
  p << 0.0;
  Eigen::Matrix2d K;
  K << 2.734952180000, 2.217356970000, 2.217356970000, 3.836806820000;
  Eigen::VectorXd g1(param_dim);
  g1 << -5.469904360000;
  Eigen::VectorXd g2(param_dim);
  g2 << -4.434713940000;
	covf = factory.create(input_dim, "CovLinearone");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, Linearard) {
  const int param_dim = input_dim + 1;
  Eigen::VectorXd p(param_dim);
  p << 0.1,0.3,-0.2,0.0,-0.1,0.0;
  Eigen::Matrix2d K;
  K << 1.0, 0.341677691433, 0.341677691433, 1.0;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 0.0, 0.0, 0.0, 0.0, 2.0;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.001917866624, 0.070600964942, 0.051675005912, 
        0.360868801414, 0.248784102634, 0.683355382866;
	covf = factory.create(input_dim, "CovLinearard");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, SEiso) {
  const int param_dim = 2;
  Eigen::VectorXd p(param_dim);
  p << -0.1, 0.1;
  Eigen::Matrix2d K;
  K << 1.221402758160, 0.331178966544, 0.331178966544, 1.221402758160;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 2.442805516320;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.864440931000, 0.662357933089;
	covf = factory.create(input_dim, "CovSEiso");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, SEard) {
  const int param_dim = input_dim + 1;
  Eigen::VectorXd p(param_dim);
  p << 0.1,0.3,-0.2,0.0,-0.1,0.0;
  Eigen::Matrix2d K;
  K << 1.0, 0.341677691433, 0.341677691433, 1.0;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 0.0, 0.0, 0.0, 0.0, 2.0;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.001917866624, 0.070600964942, 0.051675005912, 
        0.360868801414, 0.248784102634, 0.683355382866;
	covf = factory.create(input_dim, "CovSEard");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, Matern3iso) {
  const int param_dim = 2;
  Eigen::VectorXd p(param_dim);
  p << 0.1, 0.1;
  Eigen::Matrix2d K;
  K << 1.221402758160, 0.406628204830, 0.406628204830, 1.221402758160;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 2.442805516320;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.648539830522, 0.813256409659; 
  covf = factory.create(input_dim, "CovMatern3iso");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, Matern5iso) {
  const int param_dim = 2;
  Eigen::VectorXd p(param_dim);
  p << -0.2, 0.1;
  Eigen::Matrix2d K;
  K << 1.221402758160, 0.232277486841, 0.232277486841, 1.221402758160;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 2.442805516320;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.597885349117, 0.464554973682; 
  covf = factory.create(input_dim, "CovMatern5iso");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, RBFCS) {
  const int param_dim = 2;
  Eigen::VectorXd p(param_dim);
  p << -0.1, 0.1;
  Eigen::Matrix2d K;
  K << 1.221402758160, 0.331178966544, 0.331178966544, 1.221402758160;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 0.0;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.0, 0.0;
	covf = factory.create(input_dim, "CovRBFCS");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, RQiso) {
  const int param_dim = 3;
  Eigen::VectorXd p(param_dim);
  p << -0.02, 0.05, 0.2;
  Eigen::Matrix2d K;
  K << 1.105170918076, 0.501217242850, 0.501217242850, 1.105170918076;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 2.210341836151, 0.0;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.583521014075, 1.002434485700, -0.104559812648;
  covf = factory.create(input_dim, "CovRQiso");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}

TEST(CovTest, Sum) {
  const int param_dim = input_dim + 2;
  Eigen::VectorXd p(param_dim);
  p << -0.1,0.1,0.0,0.2,-0.3,0.05,-3;
  Eigen::Matrix2d K;
  K << 1.107649670252, 0.365639720307, 0.365639720307, 1.107649670252;
  Eigen::VectorXd g1(param_dim);
  g1 << 0.0, 0.0, 0.0, 0.0, 0.0, 2.210341836151, 0.004957504353;
  Eigen::VectorXd g2(param_dim);
  g2 << 0.003061772641, 0.112710706889, 0.037068028324, 0.258861991710, 
        0.397170663232, 0.731279440615, 0.0; 
	covf = factory.create(input_dim, "CovSum(CovSEard, CovNoise)");
  test_covf(param_dim, p, K, g1, g2);
  delete covf;
}
