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

#include "gp_utils.h"

namespace libgp {

  double Utils::randn() 
  {
    double u1 = 1.0 - drand48(), u2 = 1.0 - drand48();
    return sqrt(-2*log(u1))*cos(2*M_PI*u2);
  }
  
  int * Utils::randperm(int n) 
  {
    assert(n > 0);
    assert(n<RAND_MAX);
    int * array = new int[n];
    int i, j;
    double tmp;
    for (i=0; i<n; i++) array[i] = i;
    for (i=n-1; i>0; --i) {
      j = int(drand48() * (i + 1));
      tmp = array[j];
      array[j] = array[i];
      array[i] = tmp;
    }
    return array;
  }
  
  uint32_t Utils::randi(uint32_t n) 
  {
    return drand48()*n;
  }
  
  double Utils::cdf_norm (double x) 
  {
    double abs_x = fabs(x), build, norm;
    if (abs_x > 37) return norm = 0;
    else if (abs_x < 7.07106781186547) {
      build = 3.52624965998911E-02 * abs_x + 0.700383064443688;
      build = build * abs_x + 6.37396220353165;
      build = build * abs_x + 33.912866078383;
      build = build * abs_x + 112.079291497871;
      build = build * abs_x + 221.213596169931;
      build = build * abs_x + 220.206867912376;
      norm = exp(-abs_x*abs_x/2) * build;
      build = 8.83883476483184E-02 * abs_x + 1.75566716318264;
      build = build * abs_x + 16.064177579207;
      build = build * abs_x + 86.7807322029461;
      build = build * abs_x + 296.564248779674;
      build = build * abs_x + 637.333633378831;
      build = build * abs_x + 793.826512519948;
      build = build * abs_x + 440.413735824752;
      norm = norm / build;
    } else {
      build = abs_x + 0.65;
      build = abs_x + 4 / build;
      build = abs_x + 3 / build;
      build = abs_x + 2 / build;
      build = abs_x + 1 / build;
      norm = exp(-abs_x*abs_x/2) / build / 2.506628274631;
    }
    if (x > 0) norm = 1 - norm; 
    return norm;
  }
      
}
