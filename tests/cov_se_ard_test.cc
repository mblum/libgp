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

#include "cov_se_ard.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(CovSEardTest, get) {

  libgp::CovSEard cov;
  const int input_dim = 5;
  const int param_dim = input_dim + 1;
  Eigen::VectorXd p(param_dim);
  Eigen::VectorXd a(input_dim);
  Eigen::VectorXd b(input_dim);
  p << 0.8491,0.9340,0.6787,0.7577,0.7431,1.1000;
  a << 0.9143,-0.0292,0.6006,-0.7162,-0.1565;
  b << 0.8315,0.5844,0.9190,0.3115,-0.9286;

  cov.init(input_dim);  
  ASSERT_EQ(input_dim, cov.get_input_dim());
  ASSERT_EQ(param_dim, cov.get_param_dim());
  ASSERT_TRUE(cov.set_loghyper(p));
  ASSERT_NEAR(7.1980, cov.get(a, b), 0.0001);
  ASSERT_NEAR(7.1980, cov.get(b, a), 0.0001);
  ASSERT_NEAR(9.0250, cov.get(a, a), 0.0001);
  ASSERT_NEAR(9.0250, cov.get(b, b), 0.0001);
}