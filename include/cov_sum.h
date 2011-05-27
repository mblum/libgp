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

#ifndef COV_SUM_H_8M4R5HLE
#define COV_SUM_H_8M4R5HLE

#include "cov.h"

namespace libgp
{
/** Sums of covariance functions.
 * @author Manuel Blum 
 * @ingroup cov_group
 */
class CovSum : public CovarianceFunction
{
public:
	CovSum ();
	virtual ~CovSum ();
	bool init(int n, CovarianceFunction * first, CovarianceFunction * second);
	double get(Eigen::VectorXd &x1, Eigen::VectorXd &x2);
	void grad(Eigen::VectorXd &x1, Eigen::VectorXd &x2, Eigen::VectorXd &grad);
	bool set_loghyper(Eigen::VectorXd &p);
	virtual std::string to_string();
private:
	size_t param_dim_first;
	size_t param_dim_second;
	CovarianceFunction * first;
	CovarianceFunction * second;
};

}

#endif /* end of include guard: COV_SUM_H_8M4R5HLE */

