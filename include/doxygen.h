// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 
 \page licence Licensing
 
 libgp - Gaussian process library for Machine Learning
 Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
 All rights reserved.

 \verbinclude "../COPYING"
 
 \mainpage libgp - A Gaussian Process library for Machine Learning
 
 \section about About
 libgp is a C++ library for Gaussian Process regression. More information on Gaussian processes can be found in the book
 <a href="http://www.gaussianprocess.org/gpml/" target="_blank">Gaussian Processes for Machine Learning</a> by 
 Carl Edward Rasmussen and Christopher K. I. Williams which is available for 
 <a href="http://www.gaussianprocess.org/gpml/chapters">download</a> in electronic format.
 
 \section requirements Requirements
 - libgp was tested under Linux and MacOSX
 - <a href="http://www.cmake.org/" target="_blank">cmake</a>: cross-platform, open-source build system
 - <a href="http://eigen.tuxfamily.org" target="_blank">Eigen3</a>: template library for linear algebra
 - <a href="http://www.cise.ufl.edu/research/sparse/cholmod/" target="_blank">CHOLMOD</a>: supernodal sparse Cholesky factorization and update/downdate
 - <a href="http://www.netlib.org/blas/" target="_blank">BLAS</a>: Basic Linear Algebra Subprograms
 - <a href="http://code.google.com/p/googletest" target="_blank">googletest</a> (optional)
 
 \section start Getting started
 -# <a href="https://sourceforge.net/projects/libgp/files/">Download</a>  the latest version of libgp 
 and unpack the archive.
 -# Create a build directory in the project folder, run cmake and make.
 \verbatim mkdir build 
cd build
cmake ..
make\endverbatim
 -# Check out the examples on how to use the library.
 
 For support send an email to the developer: @htmlonly <script type="text/javascript">eval(unescape('d%6fc%75%6de%6e%74%2e%77%72%69%74e%28%27%3Ca%20%68%72ef%3D%22%26%23109%3Ba%26%23105%3B%6c%26%23116%3B%26%23111%3B%3A%26%23109%3B%26%2398%3B%26%23108%3B%26%23117%3B%26%23109%3B%26%2364%3B%26%23105%3B%26%23110%3B%26%23102%3B%26%23111%3B%26%23114%3B%26%23109%3B%26%2397%3B%26%23116%3B%26%23105%3B%26%23107%3B%26%2346%3B%26%23117%3B%26%23110%3B%26%23105%3B%26%2345%3B%26%23102%3B%26%23114%3B%26%23101%3B%26%23105%3B%26%2398%3B%26%23117%3B%26%23114%3B%26%23103%3B%26%2346%3B%26%23100%3B%26%23101%3B%22%3E%4da%6e%75e%6c%20B%6c%75%6d%3C%2fa%3E%27%29%3B'));</script>@endhtmlonly
 
 \section release Release Notes
 2011/09/28 version 0.1.3
 - implemented sparse Gaussian processes using Cholmod
 - improved organization of training data
 - improved interfaces
 
 2011/06/03 version 0.1.2
 - added Matern5 covariance function
 - added isotropic rational quadratic covariance function
 - added function to draw random data according to covariance function
 
 2011/05/27 version 0.1.1
 - google-tests added
 - added Matern3 covariance function
 - various bugfixes
 
 2011/05/26 version 0.1.0 
 - basic functionality for standard gp regression
 - most important covariance functions implemented
 - capability to read and write models to disk 
 
 \example gp_example_dense.cc
 \example gp_example_sparse.cc
 
 \page faq Frequently Asked Questions
 nothing so far 
 
 */
