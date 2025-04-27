from . import libgp_cpp


class OptimizerRProp(libgp_cpp.RProp):
    """RProp (Resilient Backpropagation) optimizer for training Gaussian Process models.

    This class implements the RProp optimization algorithm for training Gaussian Process models
    by maximizing the marginal likelihood. RProp is a first-order optimization method that
    adapts the step size for each parameter independently.

    Args:
        eps_stop (float, optional): Stopping criterion for convergence. Defaults to 1e-5.
        Delta0 (float, optional): Initial step size. Defaults to 0.1.
        Deltamin (float, optional): Minimum step size. Defaults to 1e-6.
        Deltamax (float, optional): Maximum step size. Defaults to 50.
        etaminus (float, optional): Decrease factor for step size. Defaults to 0.5.
        etaplus (float, optional): Increase factor for step size. Defaults to 1.2.
    """

    def __init__(
            self,
            eps_stop: float = 1e-5,
            Delta0: float = 0.1,
            Deltamin: float = 1e-6,
            Deltamax: float = 50,
            etaminus: float = 0.5,
            etaplus: float = 1.2,
    ) -> None:
        super().init(eps_stop, Delta0, Deltamin, Deltamax, etaminus, etaplus)
        super().__init__()

    def maximize(self, gp: libgp_cpp.GaussianProcess, n: int = 100, verbose: bool = False) -> None:
        """Optimize the Gaussian Process model using RProp.

        Maximizes the marginal likelihood of the Gaussian Process model using the RProp
        optimization algorithm. The algorithm iteratively updates the hyperparameters
        of the covariance function.

        Args:
            gp (libgp_cpp.GaussianProcess): The Gaussian Process model to optimize.
            n (int, optional): Maximum number of iterations. Defaults to 100.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.

        Raises:
            TypeError: If the provided model is not a Gaussian Process instance.
        """
        if not isinstance(gp, libgp_cpp.GaussianProcess):
            raise TypeError("The provided model is not a Gaussian Process.")

        super().maximize(gp, n, verbose)


class OptimizerConjugateGradient(libgp_cpp.CG):
    """Conjugate Gradient optimizer for training Gaussian Process models.

    This class implements the Conjugate Gradient optimization algorithm for training 
    Gaussian Process models by maximizing the marginal likelihood. Conjugate Gradient
    is a second-order optimization method that uses gradient information to determine
    search directions.
    """

    def __init__(self) -> None:
        super().__init__()

    def maximize(self, gp: libgp_cpp.GaussianProcess, n: int = 100, verbose: bool = False) -> None:
        """Optimize the Gaussian Process model using Conjugate Gradient.

        Maximizes the marginal likelihood of the Gaussian Process model using the
        Conjugate Gradient optimization algorithm. The algorithm iteratively updates
        the hyperparameters of the covariance function.

        Args:
            gp (libgp_cpp.GaussianProcess): The Gaussian Process model to optimize.
            n (int, optional): Maximum number of iterations. Defaults to 100.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.

        Raises:
            TypeError: If the provided model is not a Gaussian Process instance.
        """
        if not isinstance(gp, libgp_cpp.GaussianProcess):
            raise TypeError("The provided model is not a Gaussian Process.")

        super().maximize(gp, n, verbose)
