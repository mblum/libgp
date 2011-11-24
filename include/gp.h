// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __GP_H__
#define __GP_H__

#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"
#include "doxygen.h"

namespace libgp {
  
  /** Gaussian process regression.
   *  @author Manuel Blum */
  class GaussianProcess
  {
  public:
    
    /** Create and instance of GaussianProcess with given input dimensionality 
     *  and covariance function. */
    GaussianProcess (size_t input_dim, std::string covf_def);
    
    /** Create and instance of GaussianProcess from file. */
    GaussianProcess (const char * filename);
    
    virtual ~GaussianProcess ();
    
    /** Write current gp model to file. */
    void write(const char * filename);
    
    /** Update covariance matrix and perform cholesky decomposition. */
    virtual void compute();
    
    /** Predict target value for given input.
     *  @param x input vector
     *  @return predicted value */
    virtual double f(const double x[]);
    
    /** Predict variance of prediction for given input.
     *  @param x input vector
     *  @return predicted variance */
    virtual double var(const double x[]);
    
    /** Add input-output-pair to sample set.
     *  Add a copy of the given input-output-pair to sample set.
     *  @param x input array
     *  @param y output value
     */
    void add_pattern(const double x[], double y);

    /** Get number of samples in the training set. */
    size_t get_sampleset_size();
    
    /** Clear sample set and free memory. */
    void clear_sampleset();
    
    /** Get reference on currently used covariance function. */
    CovarianceFunction & covf();
    
    /** Get input vector dimensionality. */
    size_t get_input_dim();
    
  protected:
    
    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    
    /** Last query vector. */ 
    Eigen::VectorXd x_star;

    /** Last test kernel vector. */
    Eigen::VectorXd k_star;

    /** Linear solver used to invert the covariance matrix. */
    Eigen::LLT<Eigen::MatrixXd> solver;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
    /** Update test input and cache kernel vector for performance. */
    void update(const double x[]);
  };
}

#endif /* __GP_H__ */
