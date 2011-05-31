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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>
#include <cmath>

namespace libgp {

/** Generates independent standard normally distributed (zero expectation, 
 *  unit variance) random numbers using the Box–Muller transform. */
inline double randn() {
  double u1 = 1.0 - drand48(), u2 = 1.0 - drand48();
  return sqrt(-2*log(u1))*cos(2*M_PI*u2);
}

}

#endif /* __UTIL_H__ */
