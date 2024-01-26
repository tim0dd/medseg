import unittest

import torch
from torch import Tensor

from medseg.models.segmentors import FCBFormer


class FcbFormerTests(unittest.TestCase):
    def test_loss_func(self):

        # initialize model
        fcbformer = FCBFormer(in_size=512)

        # binary loss
        loss_func_binary = fcbformer.default_loss_func(multiclass=False)
        self.assertTrue(callable(loss_func_binary))

        # generate a prediction and target
        pred = Tensor(torch.rand(1, 1, 512, 512))
        target = Tensor(torch.rand(1, 1, 512, 512))
        pred.requires_grad = True
        target.requires_grad = True

        # compute loss and check properties
        loss = loss_func_binary(pred, target)
        with torch.no_grad():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertGreaterEqual(loss.item(), 0)

        # compute gradients and check for NaN and infinite values
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(pred.grad)))
        self.assertTrue(torch.all(torch.isfinite(target.grad)))

        # multiclass loss
        loss_func_multi = fcbformer.default_loss_func(multiclass=True)
        self.assertTrue(callable(loss_func_multi))

        # generate a prediction and target
        pred = Tensor(torch.rand(1, 3, 512, 512))
        target = Tensor(torch.rand(1, 3, 512, 512))
        pred.requires_grad = True
        target.requires_grad = True

        # compute loss and check properties
        loss = loss_func_multi(pred, target)
        with torch.no_grad():
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertGreaterEqual(loss.item(), 0)

        # compute gradients and check for NaN and infinite values
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(pred.grad)))
        self.assertTrue(torch.all(torch.isfinite(target.grad)))
