// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef COV_NOISE_H_UFVDC04P
#define COV_NOISE_H_UFVDC04P

#include "cov.h"

namespace libgp
{
  
  /** Independent covariance function (white noise).
   *  @author Manuel Blum
   *  @ingroup cov_group */
  class CovNoise : public CovarianceFunction
  {
  public:
    CovNoise ();
    virtual ~CovNoise ();
    bool init(int n);
    double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2);
    void grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad);
    void set_loghyper(const Eigen::VectorXd &p);
    virtual std::string to_string();
    virtual double get_threshold();
    virtual void set_threshold(double threshold);
  private:
    double s2;
  };
  
}

#endif /* end of include guard: COV_NOISE_H_UFVDC04P */
