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

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "gp_utils.h"

TEST(Utils, randn) {
  int n = 10e5;
  Eigen::VectorXd x(n);
  for(size_t k = 0; k < 10; ++k) {
    for(size_t i = 0; i < n; ++i) x(i) = libgp::randn();
    double mean = x.mean();  
    for(size_t i = 0; i < n; ++i) x(i) = x(i) - mean;
    x = x.cwiseProduct(x);
    double var = x.mean();
    ASSERT_NEAR(0.0, mean, 10e-3);
    ASSERT_NEAR(1.0, var, 10e-3);
  }
}

