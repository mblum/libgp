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

#ifndef __COV_MATERN5_ISO_H__
#define __COV_MATERN5_ISO_H__

#define SQRT5 2.23606797749978980505147774238139390945434570312500

#include "cov.h"

namespace libgp
{
/** Matern covariance function with nu = 5/2 and isotropic distance measure.
 *  @ingroup cov_group
 *  @author Manuel Blum
 */
class CovMatern5iso : public CovarianceFunction
{
public:
	CovMatern5iso ();
	virtual ~CovMatern5iso ();
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

#endif /* __COV_MATERN5_ISO_H__ */
