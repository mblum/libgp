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

namespace libgp {

/** Generate independent standard normally distributed (zero mean, 
 *  unit variance) random numbers using the Box–Muller transform. */
inline double randn() {
  double u1 = 1.0 - drand48(), u2 = 1.0 - drand48();
  return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}
  
/** Generate random permutation of array (0, ..., n-1) using 
 *  Fisher–Yates shuffle algorithm. */
inline int * randperm(int n) {
  assert(n > 0);
  assert(n<RAND_MAX);
  int * array = new int[n];
  int i, j;
  double tmp;
  for (i=0; i<n; i++) array[n] = n;
  for (i=n-1; i>0; --i) {
    j = rand() % (i+1);
    tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
  return array;
}
  
}

#endif /* __GP_UTILS_H__ */
