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

#include "sampleset.h"

namespace libgp {
  
  SampleSet::SampleSet (int input_dim)
  {
    this->input_dim = input_dim;
  }
  
  SampleSet::~SampleSet() 
  {
    clear();
  }
  
  void SampleSet::add(const double x[], double y)
  {
    Eigen::VectorXd * v = new Eigen::VectorXd(input_dim);
    for (size_t i=0; i<input_dim; ++i) (*v)(i) = x[i];
    inputs.push_back(v);
    targets.push_back(y);
    assert(inputs.size()==targets.size());
    n = inputs.size();
  }
  
  const Eigen::VectorXd & SampleSet::x(size_t k)
  {
    return *inputs.at(k);
  }
  
  double SampleSet::y(size_t k)
  {
    return targets.at(k);
  }
  
  size_t SampleSet::size()
  {
    return n;
  }
  
  void SampleSet::clear()
  {
    while (!inputs.empty()) {
      delete inputs.back();
      inputs.pop_back();
    }    
    n = 0;
    targets.clear();
  }
  
  bool SampleSet::empty ()
  {
    return n==0;
  }
}
