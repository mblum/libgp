// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "gp_utils.h"
#include "cov_factory.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

void test(std::string covstr)
{
  int n = 3;
  libgp::CovFactory factory;
  libgp::CovarianceFunction * cov = factory.create(n, covstr);
  int param_dim = cov->get_param_dim();
  Eigen::VectorXd grad = Eigen::VectorXd::Random(param_dim);
  Eigen::VectorXd params = Eigen::VectorXd::Random(param_dim);
  Eigen::VectorXd x1 = Eigen::VectorXd::Random(n);
  Eigen::VectorXd x2 = Eigen::VectorXd::Random(n);  

  cov->set_loghyper(params);
  cov->grad(x1, x2, grad);
 
  double e = 1e-4; 

  for (int i=0; i<param_dim; ++i) {
    double theta = params(i);
    params(i) = theta - e;
    cov->set_loghyper(params);
    double j1 = cov->get(x1, x2);
    params(i) = theta + e;
    cov->set_loghyper(params);
    double j2 = cov->get(x1, x2);
    params(i) = theta;

    // hack to ignore period hyperparameter of CovPeriodicMatern3iso
    if (covstr.compare("CovPeriodicMatern3iso") == 0 && i == n-1) ASSERT_NEAR(0.0, grad(i), 1e-6);
    else ASSERT_NEAR((j2-j1)/(2*e), grad(i), 1e-6);
  }
  delete cov;
}

TEST(CovGradTest, CovLinearard) 
{
  test("CovLinearard");
}


TEST(CovGradTest, CovLinearone) 
{
  test("CovLinearone");
}

TEST(CovGradTest, CovMatern3iso) 
{
  test("CovMatern3iso");
}

TEST(CovGradTest, CovMatern5iso) 
{
  test("CovMatern5iso");
}

TEST(CovGradTest, CovNoise) 
{
  test("CovNoise");
}

TEST(CovGradTest, CovRQiso) 
{
  test("CovRQiso");
}

TEST(CovGradTest, CovSEard) 
{
  test("CovSEard");
}

TEST(CovGradTest, CovSEiso) 
{
  test("CovSEiso");
}

TEST(CovGradTest, CovPeriodicMatern3iso) 
{
  test("CovPeriodicMatern3iso");
}
