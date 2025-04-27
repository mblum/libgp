import numpy as np

from . import libgp_cpp


class GaussianProcess(libgp_cpp.GaussianProcess):
    """
    Gaussian Process Regression class.

    This class provides methods for training and predicting with a Gaussian Process model.
    """

    def add_pattern(self, x: np.array, y: float) -> None:
        """Add a single training pattern to the Gaussian Process model.

        Parameters:
        - x: Input features as a numpy array.
        - y: Target value as a float.
        """
        super().add_pattern(x, y)

    def add_patterns(self, X: np.array, y: np.array) -> None:
        """Add training patterns to the Gaussian Process model.

        Parameters:
        - X: Input features as a numpy array.
        - y: Target values as a numpy array.
        """

        if len(X) != len(y):
            raise ValueError("The length of X and y must be the same.")

        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.flatten()

        super().add_patterns(X, y)

    def predict(self, X: np.array) -> np.array:
        """Predict the mean for the given input features.

        Parameters:
        - X: Input features as a numpy array.

        Returns:
        - A numpy array containing the predicted mean.
        """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        mean = super().predict(X, False)
        return mean.flatten()

    def predict_with_variance(self, X: np.array) -> tuple:
        """Predict the mean and variance for the given input features.

        Parameters:
        - X: Input features as a numpy array.

        Returns:
        - A tuple containing the predicted mean and variance.
        """

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        result = super().predict(X, True)
        return result[:, 0], result[:, 1]

    def set_y(self, y: np.array) -> None:
        """Set the target values for the Gaussian Process model.

        Parameters:
        - y: Target values as a numpy array.
        """
        super().set_y(y)

    def get_sampleset_size(self) -> int:
        """Get the size of the sample set.

        Returns:
        - The number of samples in the training set.
        """
        return super().get_sampleset_size()

    def clear_sampleset(self) -> None:
        """Clear the training set."""
        super().clear_sampleset()

    def get_log_likelihood(self) -> float:
        """Get the log likelihood of the current model.

        Returns:
        - The log likelihood as a float.
        """
        return super().get_log_likelihood()

    def get_log_likelihood_gradient(self) -> np.array:
        """Get the gradient of the log likelihood.

        Returns:
        - The gradient as a numpy array.
        """
        return super().get_log_likelihood_gradient()

    def get_input_dim(self) -> np.array:
        """Get the dimensionality of the input space.

        Returns:
        - The number of input dimensions as an integer.
        """
        return super().get_input_dim()

    def set_loghyper(self, loghyper: np.array) -> None:
        """Set the hyperparameters of the Gaussian Process model.

        Parameters:
        - loghyper: Hyperparameters as a numpy array.
        """
        super().set_loghyper(loghyper)

    def get_loghyper(self) -> np.array:
        """Get the hyperparameters of the Gaussian Process model.

        Returns:
        - The hyperparameters as a numpy array.
        """
        return super().get_loghyper()

    def get_param_dim(self) -> int:
        """Get the dimensionality of the hyperparameter space.

        Returns:
        - The number of hyperparameters as an integer.
        """
        return super().get_param_dim()
