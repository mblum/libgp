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

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace libgp {
  
  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    sampleset = new SampleSet(input_dim);
  }
  
  GaussianProcess::GaussianProcess (const char * filename)
  {
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          covf().set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
    //update = 1;
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    delete sampleset;
    delete cf;
  }  
  
  void GaussianProcess::compute()
  {
    if (sampleset->empty()) return; 
    Eigen::MatrixXd K(sampleset->size(), sampleset->size());
    alpha.resize(sampleset->size());
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        K(i, j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
      alpha(i) = sampleset->y(i);
    }
    // perform cholesky factorization
    solver.compute(K.selfadjointView<Eigen::Lower>());
    solver.solveInPlace(alpha);
  }
  
  double GaussianProcess::f(const double x[])
  {
    Eigen::VectorXd kstar(sampleset->size());
    // compute covariance between input and training data	
    Eigen::Map<const Eigen::VectorXd> x_vec_map(x, input_dim);
    for(size_t i = 0; i < sampleset->size(); ++i) {
      kstar(i) = cf->get((Eigen::VectorXd &) x_vec_map, sampleset->x(i));
    }
    // compute predicted value
    return kstar.dot(alpha);    
  }
  
  double GaussianProcess::var(const double x[])
  {
    return 0.0;
  }
  
  /*
  double GaussianProcess::predict(const double x[], double &var, bool compute_variance)
  {
    if (sampleset->size()==0) return 0.0;
    // update cached alpha if outdated
    if (update) {
      update = 0;
      Eigen::MatrixXd K(sampleset->size(), sampleset->size());
      alpha.resize(sampleset->size());
      // compute kernel matrix (lower triangle)
      for(size_t i = 0; i < sampleset->size(); ++i) {
        for(size_t j = 0; j <= i; ++j) {
          K(i, j) = covf->get(sampleset->x(i), sampleset->x(j));
        }
        alpha(i) = sampleset->y(i);
      }
      // perform cholesky factorization
      solver = K.selfadjointView<Eigen::Lower>().llt();
      solver.solveInPlace(alpha);
    }
    Eigen::VectorXd kstar(sampleset->size());
    // compute covariance between input and training data	
    Eigen::Map<const Eigen::VectorXd> x_vec_map(x, input_dim);
    for(size_t i = 0; i < sampleset->size(); ++i) {
      kstar(i) = covf->get((Eigen::VectorXd &) x_vec_map, sampleset->x(i));
    }
    // compute predicted value
    double fstar = kstar.dot(alpha);
    // compute variance
    if (compute_variance) {
      Eigen::VectorXd v = solver.matrixL().solve(kstar);
      var = covf->get((Eigen::VectorXd &) x_vec_map, (Eigen::VectorXd &) x_vec_map) - v.dot(v);	
    }
    return fstar;
  }
  
  double GaussianProcess::predict(const double x[], double &var) 
  {
    return predict(x, var, 1);
  }
  
  double GaussianProcess::predict(const double x[])
  {
    double var;
    return predict(x, var, 0);
  }*/
  
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    sampleset->add(x, y);
    //update = 1;
  }
  
  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset->clear();
  }
  
  void GaussianProcess::write(const char * filename)
  {
    // output
    std::ofstream outfile;
    outfile.open(filename);
    time_t curtime = time(0);
    tm now=*localtime(&curtime);
    char dest[BUFSIZ]= {0};
    strftime(dest, sizeof(dest)-1, "%c", &now);
    outfile << "# " << dest << std::endl << std::endl
    << "# input dimensionality" << std::endl << input_dim << std::endl 
    << std::endl << "# covariance function" << std::endl 
    << cf->to_string() << std::endl << std::endl
    << "# log-hyperparameter" << std::endl;
    Eigen::VectorXd param = cf->get_loghyper();
    for (size_t i = 0; i< cf->get_param_dim(); i++) {
      outfile << std::setprecision(10) 
      << param(i) << " ";
    }
    outfile << std::endl << std::endl 
    << "# data (target value in first column)" << std::endl;
    for (size_t i=0; i<sampleset->size(); ++i) {
      outfile << std::setprecision(10) << sampleset->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }
  
  CovarianceFunction & GaussianProcess::covf()
  {
    return *cf;
  }
}