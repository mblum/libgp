// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

namespace libgp {
  
  const double log2pi = log(2*M_PI)/2;

  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def) :
    sampleset_changed(true)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    sampleset = new SampleSet(input_dim);
  }
  
  GaussianProcess::GaussianProcess (const char * filename) :
    sampleset_changed(true)
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
          cf->set_loghyper(params);
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
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    delete sampleset;
    delete cf;
  }  
  
  void GaussianProcess::compute()
  {
    if (!sampleset_changed && !cf->loghyper_changed) return;
    Eigen::MatrixXd K(sampleset->size(), sampleset->size());
    alpha.resize(sampleset->size());
    k_star.resize(sampleset->size());
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        K(i, j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
    }
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    // perform cholesky factorization
    solver.compute(K.selfadjointView<Eigen::Lower>());
    assert(solver.info() ==Eigen::Success);
    alpha = solver.solve(y);
    sampleset_changed = false;
    cf->loghyper_changed = false;
  }
  
  double GaussianProcess::f(const double x[])
  {
    assert(!sampleset->empty());
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    update_k_star(x_star);
    return k_star.dot(alpha);    
  }
  
  double GaussianProcess::var(const double x[])
  {
    assert(!sampleset->empty());
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    update_k_star(x_star);
    Eigen::VectorXd v = solver.matrixL().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    compute();
    for(size_t i = 0; i < sampleset->size(); ++i) {
      k_star(i) = cf->get(x_star, sampleset->x(i));
    }
  }
  
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    sampleset_changed = true;
    sampleset->add(x, y);
  }
  
  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset_changed = true;
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
  
  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  double GaussianProcess::log_likelihood()
  {
    compute();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double logD = solver.vectorD().array().sqrt().log().sum();
    return -0.5*y.dot(alpha) - logD - sampleset->size()*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(sampleset->size(), sampleset->size());
    solver.solveInPlace(W);
    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }
}
