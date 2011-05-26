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
 

#ifndef GP_H_6YCSNNUG
#define GP_H_6YCSNNUG

#include "Eigen/Dense"
#include "cov.h"
#include "pattern.h"
#include "doxygen.h"

#include <vector>

namespace libgp {

/** Gaussian Process Regression.
 *  @author Manuel Blum
 *  @todo implement sparse Gaussian processes
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

	/** Add input-output-pair to sample set.
	 *  Add a copy of the given input-output-pair to sample set.
	 *  @param x input array
	 *  @param y output value
	 */
	void add_pattern(const double x[], double y);

	/** Predict target value of given input.
	 *  @param x input vector
	 *  @return function value of input vector
	 */
	double predict(const double x[]);

	/** Predict target value and variance of given input.
	 *  @param x input vector
	 *  @param var predicted variance
	 *  @return function value of input vector
	 */
  double predict(const double x[], double &var);

	/** Get number of samples in the training set. */
	size_t get_sampleset_size();

	/** Set hyperparameters of covariance function.
	 *  @param p parameter array
	 */
	void set_params(double p[]);

  /** Get number of parameters for this covariance function.
	 *  @return parameter vector dimensionality
	 */
	size_t get_param_dim();

  /** Write current model to file. */
	void write(const char * filename);

  /** Clear sampleset and free memory. */
	void clear_sampleset();

protected:

	/** Predict target value and variance of given input.
	 *  @param x input vector
	 *  @param var predicted variance
	 *  @param compute_variance if flase, computation of variance will be omitted
	 *  @return function value of input vector
	 */
  virtual double predict(const double x[], double &var, bool compute_variance);

	/** The covariance function of this Gaussian process. */
	CovarianceFunction* covf;

  /** The sampleset. */
	std::vector<Pattern*> sampleset;

  /** Alpha is cached for performance. */ 
	Eigen::VectorXd alpha;
  
  /** Linear solver used to invert covariance matrix. */
  Eigen::LLT<Eigen::MatrixXd> solver;

	/** Dimensionality n of input vectors. */
	size_t input_dim;
  
  /** True, if sampleset or hyperparameters have changed and the model has to be updated. */
	bool update;
};
}

#endif /* end of include guard: GP_H_6YCSNNUG */
