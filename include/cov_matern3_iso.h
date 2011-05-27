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

#ifndef __COV_MATERN3_ISO_H__
#define __COV_MATERN3_ISO_H__

#include "cov.h"

namespace libgp
{
/** Matern covariance function with nu = d/2 and isotropic distance measure. For
 *  d=1 the function is also known as the exponential covariance function or the 
 *  Ornstein-Uhlenbeck covariance in 1d.
 *  @ingroup cov_group
 *  @author Manuel Blum
 *  @todo implement this
 */
class CovMatern3iso : public CovarianceFunction
{
public:
	CovMatern3iso ();
	virtual ~CovMatern3iso ();
	bool init(int n);
	double get(Eigen::VectorXd &x1, Eigen::VectorXd &x2);
	void grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad);
	bool set_loghyper(Eigen::VectorXd &p);
	virtual std::string to_string();
private:
	double ell;
	double sf2;
};

}

#endif /* __COV_MATERN3_ISO_H__ */
