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
#include <vector>

namespace libgp {
  
  /** Container holding training patterns.
   *  @author Manuel Blum */
  class SampleSet
  {
  public:
    /** Constructor.
     *  @param input_dim dimensionality of input vectors */
    SampleSet (int input_dim);
    
    /** Destructor. */    
    virtual ~SampleSet();
    
    /** Add input-output pattern to sample set.
     *  @param x input array
     *  @param y target value */
    void add(const double x[], double y);
    
    /** Get input vector at index k. */
    const Eigen::VectorXd & x (size_t k);
    
    /** Get target value at index k. */
    double y (size_t k);
    
    /** Get number of samples. */
    size_t size();
    
    /** Clear sample set. */
    void clear();
    
    /** Check if sample set is empty. */
    bool empty ();
    
  private:
    
    /** Container holding input vectors. */
    std::vector<Eigen::VectorXd *> inputs;
    
    /** Container holding target values. */
    std::vector<double> targets;
    
    /** Dimensionality of input vectors. */
    size_t input_dim;
    
    /** Number of samples. */
    size_t n;
  };
}

#endif /* end of include guard: SAMPLESET_H */
