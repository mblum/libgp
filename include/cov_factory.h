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

#ifndef COV_FACTORY_H_DL5BMKEA
#define COV_FACTORY_H_DL5BMKEA

#include <iostream>

#include <map>
#include <fstream>
#include <sstream>

#include "cov.h"
#include "cov_noise.h"
#include "cov_se_ard.h"
#include "cov_se_iso.h"
#include "cov_matern3_iso.h"
#include "cov_matern5_iso.h"
#include "cov_rq_iso.h"
#include "cov_sum.h"

namespace libgp {

template <typename ClassName> CovarianceFunction * create_func()
{
	return new ClassName();
}

/** Factory class for generating instances of CovarianceFunction. 
 *  @author Manuel Blum
 */
class CovFactory
{
public:
	CovFactory () 
	{
		registry["CovNoise"] = & create_func<CovNoise>;
		registry["CovSEard"] = & create_func<CovSEard>;
		registry["CovSEiso"] = & create_func<CovSEiso>;
		registry["CovMatern3iso"] = & create_func<CovMatern3iso>;
		registry["CovMatern5iso"] = & create_func<CovMatern5iso>;
		registry["CovRQiso"] = & create_func<CovRQiso>;
		registry["CovSum"] = & create_func<CovSum>;
	}
	virtual ~CovFactory () {};
  /** Create an instance of CovarianceFunction. 
   *  @param input_dim input vector dimensionality
   *  @param key string representation of covariance function
   *  @return instance of CovarianceFunction
   */
	CovarianceFunction* create(size_t input_dim, const std::string key)
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
	/** Returns a string vector of available covariance functions. */
  std::vector<std::string> list()
  {
    std::vector<std::string> products;
    std::map<std::string , CovFactory::create_func_def>::iterator it;
    for (it = registry.begin(); it != registry.end(); ++it) {
      products.push_back((*it).first);
    }
    return products;
  }  
private:
	typedef CovarianceFunction*(*create_func_def)();
	std::map<std::string , CovFactory::create_func_def> registry;
};
}
#endif /* end of include guard: COV_FACTORY_H_DL5BMKEA */