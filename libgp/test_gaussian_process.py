import unittest

import numpy as np

from . import GaussianProcess


class TestGaussianProcess(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 2
        self.gp = GaussianProcess(self.input_dim, "CovSum(CovSEiso, CovNoise)")
        # Set initial hyperparameters (length scale, signal variance, noise)
        self.params = np.array([0.0, 0.0, -2.0])
        self.gp.set_loghyper(self.params)

        # Create some simple test data
        self.X_train = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        self.y_train = np.array([0.0, 0.5, 0.5, 1.0])

    def test_initialization(self):
        """Test proper initialization of GaussianProcess."""
        self.assertEqual(self.gp.get_input_dim(), self.input_dim)
        self.assertEqual(self.gp.get_sampleset_size(), 0)
        np.testing.assert_array_almost_equal(self.gp.get_loghyper(), self.params)

    def test_add_single_pattern(self):
        """Test adding a single training pattern."""
        x = np.array([0.0, 0.0])
        y = 0.0
        self.gp.add_pattern(x, y)
        self.assertEqual(self.gp.get_sampleset_size(), 1)

    def test_add_patterns(self):
        """Test adding multiple training patterns."""
        self.gp.add_patterns(self.X_train, self.y_train)
        self.assertEqual(self.gp.get_sampleset_size(), len(self.X_train))

    def test_add_patterns_validation(self):
        """Test validation when adding patterns."""
        X_invalid = np.array([[0.0, 0.0]])
        y_invalid = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            self.gp.add_patterns(X_invalid, y_invalid)

    def test_predict(self):
        """Test prediction functionality."""
        # Train the model
        self.gp.add_patterns(self.X_train, self.y_train)

        # Test single prediction
        X_test = np.array([0.5, 0.5])
        pred = self.gp.predict(X_test)
        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertEqual(pred.shape, (1,))

        # Test multiple predictions
        X_test_multiple = np.array([[0.5, 0.5], [0.25, 0.75]])
        preds = self.gp.predict(X_test_multiple)
        self.assertEqual(preds.shape, (2,))

    def test_predict_with_variance(self):
        """Test prediction with variance."""
        self.gp.add_patterns(self.X_train, self.y_train)
        X_test = np.array([0.5, 0.5])

        mean, variance = self.gp.predict_with_variance(X_test)
        self.assertTrue(isinstance(mean, np.ndarray))
        self.assertTrue(isinstance(variance, np.ndarray))
        self.assertTrue(np.all(variance >= 0))  # Variance should be non-negative

    def test_clear_sampleset(self):
        """Test clearing the training set."""
        self.gp.add_patterns(self.X_train, self.y_train)
        self.gp.clear_sampleset()
        self.assertEqual(self.gp.get_sampleset_size(), 0)

    def test_log_likelihood(self):
        """Test log likelihood calculation."""
        self.gp.add_patterns(self.X_train, self.y_train)
        ll = self.gp.get_log_likelihood()
        self.assertTrue(isinstance(ll, float))

    def test_log_likelihood_gradient(self):
        """Test log likelihood gradient calculation."""
        self.gp.add_patterns(self.X_train, self.y_train)
        grad = self.gp.get_log_likelihood_gradient()
        self.assertTrue(isinstance(grad, np.ndarray))
        self.assertEqual(len(grad), len(self.params))

    def test_hyperparameter_management(self):
        """Test getting and setting hyperparameters."""
        new_params = np.array([1.0, 0.5, -1.0])
        self.gp.set_loghyper(new_params)
        np.testing.assert_array_almost_equal(self.gp.get_loghyper(), new_params)
        self.assertEqual(self.gp.get_param_dim(), len(new_params))


if __name__ == '__main__':
    unittest.main()
