import os

import pytest
import torch
from medseg.config.config import load_and_parse_config
from medseg.training import schedulers
from torch.nn import Parameter
from torch.optim import AdamW


class TestSchedulers:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optim = AdamW([Parameter(torch.randn(2, 2, requires_grad=True))], lr=0.1)

    def test_cosine_annealing_lr(self):
        self.cfg = self.load_scheduler_test_config("config_test_cosine_annealing.yaml")
        scheduler = schedulers.get_scheduler(self.cfg, self.optim)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 10
        assert scheduler.eta_min == 0.0002
        assert scheduler.last_epoch == 0
        assert scheduler.verbose == True

    def test_cosine_annealing_warm_restarts(self):
        self.cfg = self.load_scheduler_test_config("config_test_cosine_annealing_warm_restarts.yaml")
        scheduler = schedulers.get_scheduler(self.cfg, self.optim)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert scheduler.T_0 == 10
        assert scheduler.T_mult == 2
        assert scheduler.eta_min == 0.0003
        assert scheduler.last_epoch == 0
        assert scheduler.verbose == False

    def test_reduce_lr_on_plateau(self):
        self.cfg = self.load_scheduler_test_config("config_test_reduce_lr_on_plateau.yaml")
        scheduler = schedulers.get_scheduler(self.cfg, self.optim)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.1
        assert scheduler.patience == 2
        assert scheduler.verbose is True
        assert scheduler.threshold == 0.0003
        assert scheduler.threshold_mode == 'rel'
        assert scheduler.cooldown == 4
        assert 0.0005 in scheduler.min_lrs and len(scheduler.min_lrs) == 1
        assert scheduler.eps == 0.0006

    def load_scheduler_test_config(self, cfg_name: str) -> dict:
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(os.path.join(current_dir_path, "scheduler_test_configs"), cfg_name)
        return load_and_parse_config(file_path)
