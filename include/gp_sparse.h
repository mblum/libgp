// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

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
