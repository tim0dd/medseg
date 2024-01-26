import unittest

from medseg.data.class_mapping import ClassMapping
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric, MetricsMethod
from medseg.evaluation.metrics_tracker import MetricsTracker
import torch


class TestMetricsTracker(unittest.TestCase):

    def setUp(self):
        self.tracked_metrics = [EvalMetric.DICE, EvalMetric.IOU]
        self.classes = [0, 1, 2]
        n_classes = 3
        bg_class = 0
        class_defs = [{"label": "background", "pixel_value": bg_class}]
        for i in range(n_classes):
            if i != bg_class:
                class_defs.append({"label": f"class{i}", "pixel_value": i})
        self.cm = ClassMapping(class_defs)
        self.split = SplitType.TRAIN
        self.micro_metrics_tracker = MetricsTracker(self.tracked_metrics, self.split, "test", self.cm)
        self.macro_metrics_tracker = MetricsTracker(self.tracked_metrics, self.split, "test", self.cm,
                                                    method=MetricsMethod.MACRO)

    def test_update_metrics_from_image(self):
        img_id = 'test_img'
        pred = torch.tensor([[0, 1], [1, 2]])
        mask = torch.tensor([[0, 1], [1, 2]])

        self.micro_metrics_tracker.update_metrics_from_image(img_id, pred, mask)

        for metric in self.tracked_metrics:
            self.assertIn(img_id, self.micro_metrics_tracker.metrics[metric])

    def test_update_metrics_from_batch(self):
        img_ids = ['test_img1', 'test_img2']
        preds = torch.tensor([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])
        masks = torch.tensor([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])

        self.micro_metrics_tracker.update_metrics_from_batch(img_ids, preds, masks)

        for metric in self.tracked_metrics:
            for img_id in img_ids:
                self.assertIn(img_id, self.micro_metrics_tracker.metrics[metric])

    def test_metrics_available_after_compute(self):
        img_id = 'test_img'
        pred = torch.tensor([[0, 1], [1, 2]])
        mask = torch.tensor([[0, 1], [1, 2]])
        self.micro_metrics_tracker.update_metrics_from_image(img_id, pred, mask)
        self.micro_metrics_tracker.compute_total_metrics()
        self.macro_metrics_tracker.update_metrics_from_image(img_id, pred, mask)
        self.macro_metrics_tracker.compute_total_metrics()
        for metric in self.tracked_metrics:
            self.assertIn(metric, self.micro_metrics_tracker.total_metrics)
            self.assertIn(metric, self.macro_metrics_tracker.total_metrics)

    def test_state_dict_and_from_dict(self):
        classes = [0, 1, 2]
        trial_name = "test_trial"
        metrics_tracker = MetricsTracker(self.tracked_metrics, self.split, trial_name, self.cm)

        num_samples = 5
        preds = torch.rand((num_samples, max(classes), 32, 32))
        masks = torch.rand((num_samples, max(classes), 32, 32))
        preds = torch.argmax(preds, dim=1)
        masks = torch.argmax(masks, dim=1)

        sample_ids = [f"sample_{i}" for i in range(num_samples)]
        metrics_tracker.update_metrics_from_batch(sample_ids, preds, masks)

        metrics_tracker.compute_total_metrics()

        state_dict = metrics_tracker.state_dict(reduce=False)
        state_dict_reduced = metrics_tracker.state_dict(reduce=True)

        # normal items
        normal_items = ["tracked_metrics", "split", "trial_name", "class_mapping", "ignore_background", "method",
                        "total_histograms", "total_metrics", "eval_object_sizes"]
        reduced_items = ["metrics", "histograms"]

        # test reconstruction from state dict
        new_mt = MetricsTracker.from_dict(state_dict)
        new_mt_reduced = MetricsTracker.from_dict(state_dict_reduced)
        for item in normal_items:
            if item in ['class_mapping', 'metrics_histograms', 'total_histograms']:
                continue  # TODO actually test equality
            else:
                self.assertEqual(getattr(new_mt, item), getattr(metrics_tracker, item))
                self.assertEqual(getattr(new_mt_reduced, item), getattr(metrics_tracker, item))
        for item in reduced_items:
            if item == 'metrics':
                metrics_dict_reduced = getattr(new_mt_reduced, item)
                for metric in new_mt.tracked_metrics:
                    # TODO test unreduced new_mt
                    self.assertEqual(len(metrics_dict_reduced[metric]), 0)
            else:
                self.assertEqual(len(getattr(new_mt_reduced, item)), 0)
                # TODO test unreduced new_mt

        for metric in self.tracked_metrics:
            for img_id in sample_ids:
                self.assertTrue(
                    new_mt.metrics[metric][img_id].equal(metrics_tracker.metrics[metric][img_id]))

        for img_id in sample_ids:
            self.assertTrue(new_mt.histograms[img_id].hist_intersect.equal(
                metrics_tracker.histograms[img_id].hist_intersect))
            self.assertTrue(
                new_mt.histograms[img_id].hist_union.equal(metrics_tracker.histograms[img_id].hist_union))
            self.assertTrue(
                new_mt.histograms[img_id].hist_pred.equal(metrics_tracker.histograms[img_id].hist_pred))
            self.assertTrue(
                new_mt.histograms[img_id].hist_mask.equal(metrics_tracker.histograms[img_id].hist_mask))

        for metric in self.tracked_metrics:
            self.assertEqual(new_mt.total_metrics[metric], metrics_tracker.total_metrics[metric])

        if __name__ == '__main__':
            unittest.main()
