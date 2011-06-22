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

#include "gp.h"
#include "gp_sparse.h"
#include "gp_utils.h"

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>


TEST(GPSparseRegressionTest, CompareToDense) {
  
  libgp::GaussianProcess gp(2, "CovSum(CovSEiso, CovNoise)");
  libgp::SparseGaussianProcess gp_sparse(2, "CovSum(CovRBFCS, CovNoise)");
  
  double params[] = {0.0, 0.0, -2};
  gp.covf().set_loghyper(params);
  gp_sparse.covf().set_loghyper(params);
  gp_sparse.covf().set_threshold(1e12);
  for (int i=0; i<500; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};
    double y = libgp::Utils::hill(x[0], x[1]);
    gp.add_pattern(x, y);
    gp_sparse.add_pattern(x, y);
  }
  gp.compute();
  gp_sparse.compute();
  double error, tss = 0.0;
  for (int i=0; i<500; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};
    error = gp.f(x) - gp_sparse.f(x);
    ASSERT_NEAR(0.0, error, 1e-9);
  }
  gp_sparse.covf().set_threshold(1);
  gp_sparse.compute();
  for (int i=0; i<500; ++i) {
    double x[] = {drand48()*4-2, drand48()*4-2};
    error = gp.f(x) - gp_sparse.f(x);
    tss += error*error;
  }
  ASSERT_GT(0.1, tss/500);
}
