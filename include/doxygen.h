// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2011, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

/*! 
 
 \page licence Licensing
 
 libgp - Gaussian process library for Machine Learning

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
 - <a href="http://code.google.com/p/googletest" target="_blank">googletest</a> (optional)
 
 \section start Getting started
 -# <a href="https://bitbucket.org/mblum/libgp/downloads">Download</a>  the latest version of libgp 
 and unpack the archive.
 -# Create a build directory in the project folder, run cmake and make.
 \verbatim mkdir build 
cd build
cmake ..
make\endverbatim
 -# Check out the examples on how to use the library.
 
 For support send an email to the developer: <a href="mailto:support@libgp.org">support@libgp.org</a>
 
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
