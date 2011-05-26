#include "gp.h"

#include "cmath"
#include <iostream>
#include <sys/time.h>

#define F(X, Y) (X * exp(-X*X - Y*Y))

using namespace libgp;

int main (int argc, char const *argv[])
{
	srand48(624);
	timeval start, end;
	std::cout << "Testing libgp ..." << std::endl;
	
	//  ------------------------ Training ------------------------
  // initialize gp 
  GaussianProcess * gp = new GaussianProcess(2, "Sum ( SEiso, Noise)");    
  // specify hyperparameters    
  double params[3] = {0, 0, -3};
  gp->set_params(params);
  // add training patterns
  for(size_t i = 0; i < 1000; ++i) {
    // randomly chosen input patterns
    double x[2] = {2*drand48(), 2*drand48()};
    // compute target value + noise
    double y = F(x[0], x[1]) + drand48()*0.06-0.03;
    // add pattern
    gp->add_pattern(x, y);
  }
  // write gp to disk and destroy
  gp->write("test.gp");
	delete gp;
  
	//  ------------------------ Prediction ------------------------
	// read from disk
  gp = new GaussianProcess("test.gp");
  // test performance
	double tss = 0;
	gettimeofday(&start, 0);
  for(size_t i = 0; i < 1000; ++i) {
    double x[2] = {2*drand48(), 2*drand48()};
    double y = F(x[0], x[1]);
    double f = gp->predict(x);
    double error = f - y;
    tss += error*error;
  }
	delete gp;
	gettimeofday(&end, 0);
	// report error
  std::cout << "tss = " << tss << std::endl;
	std::cout << "time: " << end.tv_sec - start.tv_sec - ((end.tv_usec - start.tv_usec)<0) 
	<< '.' << abs(end.tv_usec - start.tv_usec) << "s" << std::endl;
	return 0;
}