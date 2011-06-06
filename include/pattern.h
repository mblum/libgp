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

#ifndef PATTERN_H_3PYTT8K6
#define PATTERN_H_3PYTT8K6

#include <Eigen/Dense>

namespace libgp {

/** Training pattern consisting of input vector and target value. */
class Pattern
{
public:
  /** Allocate memory for a training pattern of given input dimensionality. */
	Pattern (int input_dim)
  {
    x.resize(input_dim);
  }
	virtual ~Pattern () {}
  /** Set input vector. */
	void set_input(const double * input)
  {
    for(int i = 0; i < x.size(); ++i) x(i) = input[i];
  }
  /** Set target vlaue. */
	void set_target(double target)
  {
    y = target;
  }
  /** Input vector. */
	Eigen::VectorXd x;
  /** Target value. */
	double y;
};
}

#endif /* end of include guard: PATTERN_H_3PYTT8K6 */
