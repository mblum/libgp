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

#ifndef __GP_H__
#define __GP_H__

#include <Eigen/Dense>

#include "cov.h"
#include "sampleset.h"
#include "doxygen.h"

namespace libgp {
  
  /** Gaussian Process Regression.
   *  @author Manuel Blum
   *  @todo implement hyperparameter learning
   */
  class GaussianProcess
  {
  public:
    
    /** Create and instance of GaussianProcess with given input dimensionality and covariance function. */
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
    
    /** Set hyperparameters of covariance function.
     *  @param p parameter array
     */
    //void set_params(double p[]);
    
    /** Get number of parameters for this covariance function.
     *  @return parameter vector dimensionality */
    //size_t get_param_dim();    
    
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
    
  protected:
    
    /** The covariance function of this Gaussian process. */
    CovarianceFunction * cf;
    
    /** The training sample set. */
    SampleSet * sampleset;
    
    /** Alpha is cached for performance. */ 
    Eigen::VectorXd alpha;
    
    /** Linear solver used to invert the covariance matrix. */
    Eigen::LLT<Eigen::MatrixXd> solver;
    
    /** Input vector dimensionality. */
    size_t input_dim;
    
  };
}

#endif /* __GP_H__ */
