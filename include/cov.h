/**************************************************************
libgp - Gaussian process library for Machine Learning
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

#ifndef COV_H_YTE8CCNB
#define COV_H_YTE8CCNB

#include <iostream>
#include <vector>

#include <Eigen/Dense>

namespace libgp
{

/** Covariance function base class.
 *  @author Manuel Blum
 *  @ingroup cov_group 
 *  @todo implement more covariance functions */
class CovarianceFunction
{
public:
	/** Constructor. */
	CovarianceFunction() {};

	/** Destructor. */
	virtual ~CovarianceFunction() {};

  /** Initialization method for atomic covariance functions. 
   *  @param input_dim dimensionality of the input vectors */
  virtual bool init(int input_dim) 
  { 
    return false;
  };

  /** Initialization method for compound covariance functions. 
   *  @param input_dim dimensionality of the input vectors 
   *  @param first first covariance function of compound
   *  @param second second covariance function of compound */
	virtual bool init(int input_dim, CovarianceFunction * first, CovarianceFunction * second)
	{
    return false;
	};

	/** Computes the covariance of two input vectors.
	 *  @param x1 first input vector
	 *  @param x2 second input vector
   *  @return covariance of x1 and x2 */
	virtual double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) = 0;

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	 *  @param x1 first input vector
	 *  @param x2 second input vector
	 *  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad) = 0;
  
	/** Update parameter vector.
	 *  @param p new parameter vector */
	virtual void set_loghyper(const Eigen::VectorXd &p);
  
	/** Update parameter vector.
	 *  @param p new parameter vector */
	virtual void set_loghyper(const double p[]);

	/** Get number of parameters for this covariance function.
	 *  @return parameter vector dimensionality */
	size_t get_param_dim();

	/** Get input dimensionality.
	 *  @return input dimensionality */
	size_t get_input_dim();

	/** Get log-hyperparameter of covariance function.
	 *  @return log-hyperparameter */
	Eigen::VectorXd get_loghyper();

  /** Returns a string representation of this covariance function.
   *  @return string containing the name of this covariance function */
	virtual std::string to_string() = 0;

	/** Draw random target values from this covariance function for input X. */
  Eigen::VectorXd draw_random_sample(Eigen::MatrixXd &X);
  
  /** Get distance threshold of this covariance function. */
  virtual double get_threshold();
  
  /** Set distance threshold of this covariance function. */
  virtual void set_threshold(double threshold);
	
protected:
	/** Input dimensionality. */
	size_t input_dim;

	/** Size of parameter vector. */
	size_t param_dim;

	/** Parameter vector containing the log hyperparameters of the covariance function.
	 *  The number of necessary parameters is given in param_dim. */
	Eigen::VectorXd loghyper;
};

}

#endif /* end of include guard: COV_H_YTE8CCNB */


/** Covariance functions available for Gaussian process models. 
 *  There are atomic and composite covariance functions. 
 *  @defgroup cov_group Covariance Functions */
