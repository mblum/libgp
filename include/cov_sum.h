// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef COV_SUM_H_8M4R5HLE
#define COV_SUM_H_8M4R5HLE

#include "cov.h"

namespace libgp
{
  /** Sums of covariance functions.
   *  @author Manuel Blum 
   *  @ingroup cov_group */
  class CovSum : public CovarianceFunction
  {
  public:
    CovSum ();
    virtual ~CovSum ();
    bool init(int n, CovarianceFunction * first, CovarianceFunction * second);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
    virtual double get_threshold();
    virtual void set_threshold(double threshold);
  private:
    size_t param_dim_first;
    size_t param_dim_second;
    CovarianceFunction * first;
    CovarianceFunction * second;
  };
  
}

#endif /* end of include guard: COV_SUM_H_8M4R5HLE */

