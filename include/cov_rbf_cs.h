// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __COV_RBF_CS_H__
#define __COV_RBF_CS_H__

#include "cov.h"

namespace libgp
{
  
  /** Radial basis covariance function with compact support. 
   *  @author Manuel Blum
   *  @ingroup cov_group */
  class CovRBFCS : public CovarianceFunction
  {
  public:
    CovRBFCS ();
    virtual ~CovRBFCS ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
    virtual double get_threshold();
    virtual void set_threshold(double threshold);
  private:
    double ell;
    double sf2;
    double threshold;
    double nu;
  };
  
}

#endif /* __COV_RBF_CS_H__ */