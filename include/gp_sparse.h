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

#ifndef __GP_SPARSE_H__
#define __GP_SPARSE_H__

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <Eigen/Sparse>
#include <unsupported/Eigen/CholmodSupport>

#include "gp.h"

namespace libgp {
  
  /** Sparse Gaussian process regression.
   *  @author Manuel Blum */
  class SparseGaussianProcess : public GaussianProcess
  {
  public:
    
    /** Create and instance of SparseGaussianProcess with given input dimensionality and covariance function. */
    SparseGaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create and instance of SparseGaussianProcess from file. */
    SparseGaussianProcess (const char * filename);
    
    virtual ~SparseGaussianProcess ();
    
    virtual void compute();

  protected:
    
    Eigen::SparseLLT<Eigen::SparseMatrix<double>, Eigen::Cholmod> solver;
    
  };
}

#endif /* __GP_SPARSE_H__ */
