import unittest

from medseg.models.model_builder import build_model
from medseg.models.segmentors import FCBFormer
from medseg.models.segmentors.segmentor import Segmentor


class ModelBuilderTests(unittest.TestCase):
    def test_build_segmentor(self):
        cfg = {"architecture": {"arch_type": "FCBFormer", "input_size": 512}}
        model = build_model(cfg, out_channels=1)
        self.assertTrue(isinstance(model, Segmentor))
        self.assertTrue(isinstance(model, FCBFormer))
