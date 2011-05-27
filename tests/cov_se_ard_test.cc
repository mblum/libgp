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

#include "cov_factory.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(CovSEardTest, get) {

	libgp::CovFactory factory;
	libgp::CovarianceFunction * covf;
  const int input_dim = 5;
  const int param_dim = input_dim + 1;
  Eigen::VectorXd p(param_dim);
  Eigen::VectorXd a(input_dim);
  Eigen::VectorXd b(input_dim);
  Eigen::VectorXd g(param_dim);
  p << 0.8491,0.9340,0.6787,0.7577,0.7431,1.1000;
  a << 0.9143,-0.0292,0.6006,-0.7162,-0.1565;
  b << 0.8315,0.5844,0.9190,0.3115,-0.9286;

	covf = factory.create(input_dim, "CovSEard");

  ASSERT_EQ(input_dim, covf->get_input_dim());
  ASSERT_EQ(param_dim, covf->get_param_dim());
  ASSERT_TRUE(covf->set_loghyper(p));
  ASSERT_NEAR(7.1980, covf->get(a, b), 0.0001);
  ASSERT_NEAR(9.0250, covf->get(a, a), 0.0001);
  ASSERT_NEAR(covf->get(a, b), covf->get(b, a), 0.0001);
  ASSERT_NEAR(covf->get(a, a), covf->get(b, b), 0.0001);
 
  covf->grad(a, a, g);
  for(size_t i = 0; i < 5; ++i) ASSERT_NEAR(0.0, g(i), 0.0001);
  ASSERT_NEAR(18.0500, g(5), 0.0001);
  covf->grad(a, b, g);
  ASSERT_NEAR(0.0090, g(0), 0.0001);
  ASSERT_NEAR(0.4186, g(1), 0.0001);
  ASSERT_NEAR(0.1878, g(2), 0.0001);
  ASSERT_NEAR(1.6704, g(3), 0.0001);
  ASSERT_NEAR(0.9707, g(4), 0.0001);
  ASSERT_NEAR(14.3958, g(5), 0.0001);
}