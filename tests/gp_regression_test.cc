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

#define F(X, Y) (X * exp(-X*X - Y*Y))

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "gp.h"
#include "gp_sparse.h"
#include "cov_factory.h"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include "gp_utils.h"
#include <Eigen/Sparse>
#include <unsupported/Eigen/CholmodSupport>

void test_gp_regression(int n, std::string covf_str, int input_dim, double params[])
{
  libgp::CovFactory factory;
  libgp::CovarianceFunction * covf = factory.create(input_dim, covf_str);
  Eigen::MatrixXd X(n, input_dim);
  X.setRandom();
  Eigen::VectorXd p(covf->get_param_dim());
  for(size_t i = 0; i < covf->get_param_dim(); ++i) p(i) = params[i];
  covf->set_loghyper(p);
  Eigen::VectorXd y = covf->draw_random_sample(X);
  libgp::GaussianProcess gp(input_dim, covf_str);    
  gp.covf().set_loghyper(p);
  for(size_t i = 0; i < n*0.8; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    gp.add_pattern(x, y(i));
  }
	double tss = 0;
  gp.compute();
  for(int i = n*0.8+1; i < n; ++i) {
    double x[input_dim];
    for(int j = 0; j < input_dim; ++j) x[j] = X(i,j);
    double f = gp.f(x);
    double error = f - y(i);
    tss += error*error;
  }
  std::cout << tss << std::endl;
  ASSERT_LE(tss/(n*0.2-1), 10e-3);
}

TEST(GPRegressionTest, SEiso) {
  std::string covf_str("CovSum ( CovSEiso, CovNoise)");
  double params[] = {0, 0, -2.3};
  test_gp_regression(200, covf_str, 2, params);
  test_gp_regression(500, covf_str, 2, params);
  test_gp_regression(800, covf_str, 3, params);
  test_gp_regression(1000, covf_str, 3, params);
}

TEST(GPRegressionTest, Matern3iso) {
  std::string covf_str("CovSum ( CovMatern3iso, CovNoise)");
  double params[] = {0, 0, -3};
  test_gp_regression(200, covf_str, 2, params);
  test_gp_regression(500, covf_str, 2, params);
  test_gp_regression(1200, covf_str, 3, params);
}

TEST(GPRegressionTest, Friedman) {
  std::string covf_str("CovSum ( CovSEiso, CovNoise)");
  Eigen::VectorXd params(3);
  params << -1.0, 0.0, -1.0;
  size_t input_dim = 10, n = 1000, m = 1000;
  libgp::GaussianProcess gp(input_dim, covf_str);    
  gp.covf().set_loghyper(params);
  for(size_t i = 0; i < n; ++i) {
    double x[input_dim];    
    for(size_t j = 0; j < input_dim; ++j) x[j] = drand48()*2-1;
    gp.add_pattern(x, libgp::Utils::friedman(x) + libgp::Utils::randn());
  }
  double tss = 0.0;
  gp.compute();
  for(size_t i = 0; i < m; ++i) {
    double x[input_dim];    
    for(size_t j = 0; j < input_dim; ++j) x[j] = drand48()*2-1;
    double error = libgp::Utils::friedman(x) - gp.f(x);
    tss += error*error;    
  }
  ASSERT_GE(15, tss/m);
}

TEST(GPRegressionTest, Sparse) {
  
  int n=500, input_dim = 2;
  libgp::GaussianProcess gp(input_dim, "CovSum(CovSEiso, CovNoise)");
  Eigen::VectorXd params(3);
  params << 0.0, 0.0, -2.3;
  gp.covf().set_loghyper(params);
  for (int i=0; i<n; ++i) {
    double x[input_dim];
    for (int j=0; j<input_dim; ++j) x[j] = drand48()*2-1;
    gp.add_pattern(x, x[0]*x[0]+x[1]*x[1]);
  }
  gp.compute();
  double tss = 0.0;
  for (int i=0; i<n; ++i) {
    double x[input_dim];
    for (int j=0; j<input_dim; ++j) x[j] = drand48()*2-1;
    double error = gp.f(x) - x[0]*x[0]+x[1]*x[1];
    tss += error*error;
  }
  std::cout << tss << std::endl;
}

double sigma = 0.12;
int n=20000, input_dim = 4;

TEST(GPRegressionTest, Dense) {
  
  libgp::SampleSet * sampleset = new libgp::SampleSet(input_dim);
  for (int i=0; i<n; ++i) {
    double x[input_dim];
    for (int j=0; j<input_dim; ++j) x[j] = drand48()*2-1;
    sampleset->add(x, libgp::Utils::randn());
  }
  Eigen::MatrixXd K(n,n);
  Eigen::VectorXd y(n);
  for (int i=0; i<n; ++i) {
    for (int j=0; j<=i; ++j) {
      K(i,j) = exp(-(sampleset->x(i)-sampleset->x(j)).squaredNorm()/(sigma*sigma))
      * std::max(0.0, pow(1-(sampleset->x(i)-sampleset->x(j)).squaredNorm()/(3*sigma), 5));
    }
    y(i) = sampleset->y(i);
  }
  std::cout << 0 << std::endl;
  Eigen::LLT<Eigen::MatrixXd> solver = K.selfadjointView<Eigen::Lower>().llt();
  solver.solveInPlace(y);
}

TEST(GPRegressionTest, SparseEx) {
 
  libgp::SampleSet * sampleset = new libgp::SampleSet(input_dim);
  for (int i=0; i<n; ++i) {
    double x[input_dim];
    for (int j=0; j<input_dim; ++j) x[j] = drand48()*2-1;
    sampleset->add(x, libgp::Utils::randn());
  }
  Eigen::SparseMatrix<double> K(n,n);
  Eigen::VectorXd y(n);
  for (int i=0; i<n; ++i) {
    K.startVec(i);
    for (int j=i; j<n; ++j) {
      double sqrtnorm = (sampleset->x(i)-sampleset->x(j)).squaredNorm();
      if (sqrtnorm<3*sigma) K.insertBack(j,i) = pow(1-sqrtnorm/(3*sigma), 5)*exp(-(sampleset->x(i)-sampleset->x(j)).squaredNorm()/(sigma*sigma));
    }    
    y(i) = sampleset->y(i);
  }
  K.finalize();
  std::cout << K.nonZeros() << std::endl;
  std::cout << 1.0 * K.nonZeros() / (n*n) << std::endl;
  //  Eigen::SparseLLT<Eigen::SparseMatrix<double> > solver;
  Eigen::SparseLLT<Eigen::SparseMatrix<double>, Eigen::Cholmod > solver(Eigen::SupernodalMultifrontal);
  solver.compute(K);
  solver.solveInPlace(y);
  if (solver.succeeded()) std::cout << "success" << std::endl;
  std::cout << y.squaredNorm() << std::endl;
}




