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

#ifndef __GP_UTILS_H__
#define __GP_UTILS_H__

#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <stdint.h>

namespace libgp {
  
  void srand_mt(uint32_t s);
  
  uint32_t rand_mt();
  
  /** Generate independent standard normally distributed (zero mean, 
   *  unit variance) random numbers using the Box–Muller transform. */
  double randn();
  
  /** Generate random permutation of array 0, ..., n-1 using 
   *  Fisher–Yates shuffle algorithm. */
  int * randperm(int n);
  
  /** Pseudorandom integers from a uniform discrete distribution.
   *  Returns a random integer drawn from the discrete uniform 
   *  distribution on 0, ..., n-1. */
  uint32_t randi(uint32_t n);

  /** Numerical approximation to the Standard Normal Cumulative 
   *  Distribution Function. */
  double cdf_norm (double x);

}

#endif /* __GP_UTILS_H__ */
