// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

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
