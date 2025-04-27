import numpy as np

from . import libgp_cpp


class GaussianProcess(libgp_cpp.GaussianProcess):
    """
    Gaussian Process Regression class.

    This class provides methods for training and predicting with a Gaussian Process model.
    """

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
