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

#ifndef COV_SE_ISO_H_IDHZLMRC
#define COV_SE_ISO_H_IDHZLMRC

#include "cov.h"

namespace libgp
{

/** Squared exponential covariance function with isotropic distance measure.
 *  Computes the squared exponential covariance
 *  \f$k_{SE}(x, y) := \alpha^2 \exp(-\frac{1}{2}(x-y)^T\Lambda^{-1}(x-y))\f$,
 *  with \f$\Lambda = diag(l^2, \dots, l^2)\f$ being the characteristic
 *  length scale and \f$\alpha\f$ describing the variability of the latent
 *  function. The parameters \f$l^2, \alpha\f$ are expected
 *  in this order in the parameter array.
 *  @ingroup cov_group
 *  @author Manuel Blum
 */
class CovSEiso : public CovarianceFunction
{
public:
	CovSEiso ();
	virtual ~CovSEiso ();
	bool init(int n);
	double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
	void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
	void set_loghyper(const Eigen::VectorXd &p);
	virtual std::string to_string();
private:
	double ell;
	double sf2;
};

}

#endif /* end of include guard: COV_SE_ISO_H_IDHZLMRC */
