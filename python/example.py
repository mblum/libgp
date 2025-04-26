import math

from pygp import GaussianProcess


def main():
    # Create a 2D Gaussian Process with SE kernel
    input_dim = 2
    gp = GaussianProcess(input_dim, "CovSum(CovSEiso, CovNoise)")

    # Set hyperparameters (log-values for length-scale, signal variance, and noise)
    params = [0.0, 0.0, -2.0]  # [log(l), log(sf), log(sigma_n)]
    gp.set_loghyper(params)

    # Generate some sample data
    x = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    y = [0.0, 0.5, 0.5, 1.0]

    # Add training data
    for xi, yi in zip(x, y):
        gp.add_pattern(xi, yi)

    # Make predictions
    test_point = [0.5, 0.5]
    mean = gp.predict(test_point)
    var = gp.get_variance(test_point)

    print(f"Number of training points: {gp.get_sampleset_size()}")
    print(f"Prediction at {test_point}:")
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {var:.3f}")
    print(f"95% Confidence Interval: [{mean-2*math.sqrt(var):.3f}, {mean+2*math.sqrt(var):.3f}]")
    print(f"Log likelihood: {gp.get_log_likelihood():.3f}")


if __name__ == "__main__":
    main()
