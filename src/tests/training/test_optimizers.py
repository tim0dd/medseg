import os
import unittest
from typing import Iterator

from torch.nn.parameter import Parameter
from torch.optim import *

from medseg.config.config import load_and_parse_config
from medseg.training.optimizers import get_optimizer


class OptimizerTests(unittest.TestCase):
    """
    Test the optimizer factory function. Tests if values from the config are parsed correctly and
    if the correct optimizer class is returned.
    """

    def test_adadelta(self):
        cfg = self.load_optim_test_config("config_test_adadelta.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, Adadelta)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["rho"] == 0.2)
            self.assertTrue(param_group["eps"] == 0.3)
            self.assertTrue(param_group["weight_decay"] == 0.4)

    def test_adagrad(self):
        cfg = self.load_optim_test_config("config_test_adagrad.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, Adagrad)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["lr_decay"] == 0.2)
            self.assertTrue(param_group["weight_decay"] == 0.3)
            self.assertTrue(param_group["initial_accumulator_value"] == 0.4)
            self.assertTrue(param_group["eps"] == 0.5)

    def test_adam(self):
        cfg = self.load_optim_test_config("config_test_adam.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, Adam)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["betas"] == (0.3, 0.2))
            self.assertTrue(param_group["eps"] == 0.5)
            self.assertTrue(param_group["weight_decay"] == 0.6)
            self.assertTrue(param_group["amsgrad"] is True)

    def test_adamax(self):
        cfg = self.load_optim_test_config("config_test_adamax.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, Adamax)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["betas"] == (0.3, 0.2))
            self.assertTrue(param_group["eps"] == 0.5)
            self.assertTrue(param_group["weight_decay"] == 0.6)

    def test_adamw(self):
        cfg = self.load_optim_test_config("config_test_adamw.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, AdamW)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["betas"] == (0.3, 0.2))
            self.assertTrue(param_group["eps"] == 0.5)
            self.assertTrue(param_group["weight_decay"] == 0.6)
            self.assertTrue(param_group["amsgrad"] is True)

    def test_asgd(self):
        cfg = self.load_optim_test_config("config_test_asgd.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, ASGD)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["lambd"] == 0.2)
            self.assertTrue(param_group["alpha"] == 0.3)
            self.assertTrue(param_group["t0"] == 4000)
            self.assertTrue(param_group["weight_decay"] == 0.5)

    def test_lbfgs(self):
        cfg = self.load_optim_test_config("config_test_lbfgs.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, LBFGS)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["max_iter"] == 20)
            self.assertTrue(param_group["max_eval"] == 30)
            self.assertTrue(param_group["tolerance_grad"] == 0.4)
            self.assertTrue(param_group["tolerance_change"] == 0.5)
            self.assertTrue(param_group["history_size"] == 60)
            self.assertTrue(param_group["line_search_fn"] == "strong_wolfe")

    def test_nadam(self):
        cfg = self.load_optim_test_config("config_test_nadam.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, NAdam)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["betas"] == (0.3, 0.2))
            self.assertTrue(param_group["eps"] == 0.5)
            self.assertTrue(param_group["weight_decay"] == 0.6)
            self.assertTrue(param_group["momentum_decay"] == 0.7)

    def test_rmsprop(self):
        cfg = self.load_optim_test_config("config_test_rmsprop.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, RMSprop)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["alpha"] == 0.2)
            self.assertTrue(param_group["eps"] == 0.3)
            self.assertTrue(param_group["weight_decay"] == 0.4)
            self.assertTrue(param_group["momentum"] == 0.5)
            self.assertTrue(param_group["centered"] is True)

    def test_rprop(self):
        cfg = self.load_optim_test_config("config_test_rprop.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, Rprop)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["etas"] == (0.2, 3))
            self.assertTrue(param_group["step_sizes"] == (0.4, 50))

    def test_sgd(self):
        cfg = self.load_optim_test_config("config_test_sgd.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, SGD)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["momentum"] == 0.2)
            self.assertTrue(param_group["dampening"] == 0)
            self.assertTrue(param_group["weight_decay"] == 0.4)
            self.assertTrue(param_group["nesterov"] is True)

    def test_sparseadam(self):
        cfg = self.load_optim_test_config("config_test_sparseadam.yaml")
        optim = get_optimizer(cfg, self.get_mock_param_iter())
        self.assertIsInstance(optim, SparseAdam)
        for param_group in optim.param_groups:
            self.assertTrue(param_group["lr"] == 0.1)
            self.assertTrue(param_group["betas"] == (0.3, 0.2))
            self.assertTrue(param_group["eps"] == 0.4)

    def get_mock_param_iter(self) -> Iterator[Parameter]:
        mock_param = Parameter(requires_grad=True)
        return iter([mock_param])

    def load_optim_test_config(self, cfg_name: str) -> dict:
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(os.path.join(current_dir_path, "optim_test_configs"), cfg_name)
        return load_and_parse_config(file_path)
