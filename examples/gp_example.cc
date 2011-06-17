#include "gp.h"
#include "gp_sparse.h"
#include "cov_factory.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <sys/time.h>

using namespace libgp;

int main (int argc, char const *argv[])
{
	srand48(15);
	timeval start, end;
  int n=1000;
	std::cout << "Testing libgp ..." << std::endl;
	
	//  ------------------- Generate sampleset -------------------
  CovFactory factory;
  CovarianceFunction * covf = factory.create(2, "CovSum ( CovSEiso, CovNoise)");
  Eigen::MatrixXd X(n, 2);
  X.setRandom();
  X *= 5;
  Eigen::VectorXd p(3);
  p << 0.0,0.0,-2.3;
  Eigen::VectorXd y(n);
  covf->set_loghyper(p);
  y = covf->draw_random_sample(X);
	
	//  ------------------------ Training ------------------------
  // initialize gp 
  GaussianProcess * gp = new GaussianProcess(2, "CovSum ( CovRBFCS, CovNoise)");    
  gp->covf().set_loghyper(p);
  // add training patterns
  for(size_t i = 0; i < n*0.8; ++i) {
    double x[2] = {X(i,0), X(i,1)};
    gp->add_pattern(x, y(i));
  }
  // write gp to disk and destroy
  gp->write("test.gp");
	delete gp;
  
	//  ------------------------ Prediction ------------------------
	// read from disk
  gp = new SparseGaussianProcess("test.gp");
  // test performance
	gettimeofday(&start, 0);
  gp->compute();
	gettimeofday(&end, 0);
	double tss = 0;
  for(int i = n*0.8+1; i < n; ++i) {
    double x[2] = {X(i,0), X(i,1)};
    double f = gp->f(x);
    double error = f - y(i);
    tss += error*error;
  }
	delete gp;
	// report error
  std::cout << "tss = " << tss << std::endl;
	std::cout << "time: " << end.tv_sec - start.tv_sec - ((end.tv_usec - start.tv_usec)<0) 
	<< '.' << abs(end.tv_usec - start.tv_usec) << "s" << std::endl;
	return 0;
}