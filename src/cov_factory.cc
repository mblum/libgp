// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_factory.h"

#include "cov_noise.h"
#include "cov_se_ard.h"
#include "cov_se_iso.h"
#include "cov_rbf_cs.h"
#include "cov_matern3_iso.h"
#include "cov_matern5_iso.h"
#include "cov_rq_iso.h"
#include "cov_sum.h"

namespace libgp {
  
  CovFactory::CovFactory () 
  {
    registry["CovNoise"] = & create_func<CovNoise>;
    registry["CovSEard"] = & create_func<CovSEard>;
    registry["CovSEiso"] = & create_func<CovSEiso>;
    registry["CovRBFCS"] = & create_func<CovRBFCS>;
    registry["CovMatern3iso"] = & create_func<CovMatern3iso>;
    registry["CovMatern5iso"] = & create_func<CovMatern5iso>;
    registry["CovRQiso"] = & create_func<CovRQiso>;
    registry["CovSum"] = & create_func<CovSum>;
  }
  CovFactory::~CovFactory () {};
  
  CovarianceFunction* CovFactory::create(size_t input_dim, const std::string key)
  {
    CovarianceFunction * covf;
    std::stringstream is(key);
    std::stringstream os(std::stringstream::out);
    std::stringstream os1(std::stringstream::out);
    std::stringstream os2(std::stringstream::out);
    char c;
    int i = 0, j = 0;
    while (is >> c) {
      if (c == '(') i++;
      else if (c == ')') i--;
      else if (c == ',') j++;
      else {
        if (i == 0) os << c;
        else if (j == 0) os1 << c;
        else os2 << c;
      }
    }
    std::map<std::string , CovFactory::create_func_def>::iterator it = registry.find(os.str());
    if (it == registry.end()) {
      std::cerr << "fatal error while parsing covariance function: " << os.str() << " not found" << std::endl;
      exit(0);
    } 
    covf = registry.find(os.str())->second();
    if (os1.str().length() == 0 && os2.str().length() == 0) {
      covf->init(input_dim);
    } else {
      covf->init(input_dim, create(input_dim, os1.str()), create(input_dim, os2.str()));
    }
    return covf;
  }
  std::vector<std::string> CovFactory::list()
  {
    std::vector<std::string> products;
    std::map<std::string , CovFactory::create_func_def>::iterator it;
    for (it = registry.begin(); it != registry.end(); ++it) {
      products.push_back((*it).first);
    }
    return products;
  }  
}