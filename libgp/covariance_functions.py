
from . import CovFactory


class CovSEiso:
    """
    Covariance function: isotropic squared exponential
    """

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.name = "CovSEiso"
        self.cov_type = CovFactory.CovSEiso
        self.hyperparameters = [0.0, 0.0]  # [log(length_scale), log(signal_variance)]
