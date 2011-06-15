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

#ifndef SAMPLESET_H
#define SAMPLESET_H

#include <Eigen/Dense>

namespace libgp {

/** 
 *  @author Manuel Blum */
class SampleSet
{
public:
  /** Constructor. 
   *  @param input_dim input dimensionality */
	SampleSet (int input_dim)
  {
    this->input_dim = input_dim;
  }
	virtual ~SampleSet() 
  {
    clear();
  }
  /** Add training pattern. 
   *  @param x input array
   *  @param y target value */
  void add(const double x[], double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  /** Get input vector.
   *  @param k index
   *  @return k-th input vector */
  const Eigen::VectorXd & x (size_t k)
  {
    return *inputs.at(k);
  }
  /** Get target value.
   *  @param k index
   *  @return k-th target value */
  double y (size_t k)
  {
    return targets.at(k);
  }
  /** Get sampleset size. */
  size_t size()
  {
    return n;
  }
  /** Clear sampleset. */
  void clear()
  {
    while (!inputs.empty()) {
      delete inputs.back();
      inputs.pop_back();
    }    
    n = 0;
    targets.clear();
  }
private:
  std::vector<Eigen::VectorXd *> inputs;
  std::vector<double> targets;
  size_t input_dim;
  size_t n;
};
}

#endif /* end of include guard: SAMPLESET_H */
