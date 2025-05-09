# libgp

![CI](https://github.com/mblum/libgp/workflows/CI/badge.svg)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<img src="assets/logo300.png" alt="libgp logo" />

A C++ library for Gaussian process regression. A Gaussian process defines a distribution over functions and inference takes place directly in function space. It is fully specified by a mean function and a positive definite covariance function.

The library supports online learning, meaning that you can add new training data without having to re-train the model from scratch. This is particularly useful for applications where new data arrives continuously and inference needs to be updated in real-time, such as in robotics, time series analysis or reinforcement learning. Online learning is achieved through caching of the Cholesky decomposition of the covariance matrix, and adding rows and columns to the matrix as new data points arrive. The computational costs of online
learning are significantly lower than re-training the model from scratch, but still
higher than learning the model in batch mode.

<img src="assets/runtime.png" alt="runtime" />

## Requirements

* C++17 compiler
* [CMake](https://cmake.org/) >= 3.14
* [Eigen3](https://eigen.tuxfamily.org/) >= 3.0.1

## Building and Testing

### Building the Library

1. Create a build directory and configure the project with tests enabled:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

2. Build the library:
```bash
cmake --build build
```

For development and debugging, you can use Debug build type:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLIBGP_BUILD_TESTS=ON
```

### Running Tests

The project uses Google Test for unit testing. Tests are automatically configured when building the project.

1. Build the tests:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLIBGP_BUILD_TESTS=ON 
cmake --build build
```

2. Run all tests:
```bash
cmake --build build --target test
```

### Examples

The library includes example code demonstrating how to use Gaussian Process regression. To build and run the examples:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLIBGP_BUILD_EXAMPLES=ON
cmake --build build
```

Then run the example:
```bash
./build/gp_example_dense 
```

The example demonstrates:
- Creating a 2D Gaussian Process with squared exponential covariance and noise
- Setting hyperparameters
- Training on sample points
- Making predictions and computing mean squared error

For more details, see the source code in `examples/gp_example_dense.cc`.


## Python Bindings

This library provides Python bindings for Gaussian Process regression. The bindings are generated using pybind11, allowing you to use the C++ library directly in Python.

```bash
pip install .
python examples/python_example.py
```

You can also build and install the Python package using pip directly from github:

```bash
pip install git+https://github.com/mblum/libgp.git
```

Check the Jupyter notebook in the `examples` directory for usage examples. 

The unit tests are also available in Python. You can run them using pytest:

```bash
python -m unittest -v
```

## Implemented covariance functions

### Simple covariance functions

* Linear covariance function.
* Linear covariance function with automatic relevance detection. 
* Matern covariance function with nu=1.5 and isotropic distance measure.
* Matern covariance function with nu=2.5 and isotropic distance measure.
* Independent covariance function (white noise).
* Radial basis covariance function with compact support.
* Isotropic rational quadratic covariance function. 
* Squared exponential covariance function with automatic relevance detection.
* Squared exponential covariance function with isotropic distance measure.

### Composite covariance functions

* Sums of covariance functions.

### Mean function

* The mean function is fixed to zero.

## Training a model

Initialize the model by specifying the input dimensionality and the covariance function.

    GaussianProcess gp(2, "CovSum ( CovSEiso, CovNoise)");

Set log-hyperparameter of the covariance function (see the Doxygen documentation, parameters should be given in order as listed).

    gp.covf().set_loghyper(params);

Add data to the training set. Input vectors x must be provided as double[] and targets y as double.

    gp.add_pattern(x, y);

Patterns can also be added in batches. The input matrix is a 2D array of double, where each row is a pattern and each column is a dimension.

    gp.add_patterns(X, y);

Predict value or variance of an input vector x. 

    f = gp.f(x);
    v = gp.var(x);

Batch inference is also supported. The input matrix is a 2D array of double, where each row is a pattern and each column is a dimension.

    f = gp.predict(X);
    f, v = gp.predict(X, compute_variance=true);

## Read and write

Use write function to save a Gaussian process model and the complete training set to a file.

    void write(const char * filename);

A new instance of the Gaussian process can be instantiated from this file using the following constructor.

    GaussianProcess (const char * filename);

## Advanced topics

* hyper-parameter optimization
* custom covariance functions
* the libgp file format

### Hyper-parameter optimization

This library contains two methods for hyper-parameter optimization; the conjugate
gradient method, and Rprop (resilient backpropagation). We recommend using Rprop.

For an example of how to call the optimizers, see `test_optimizer.cc`

Reasons for using Rprop can be found in Blum & Riedmiller (2013),
Optimization of Gaussian Process Hyperparameters using Rprop, *European Symposium
on Artificial Neural Networks*, Computational Intelligence and Learning.
