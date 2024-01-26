import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from torch import Tensor

from medseg.data.class_mapping import ClassMapping
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric
from medseg.evaluation.metrics import compute_histograms, compute_metrics
from medseg.evaluation.metrics_tracker import MetricsTracker


class MetricsTest(unittest.TestCase):

    def get_class_mapping(self, n_classes: int, bg_class: int = 0) -> ClassMapping:
        class_defs = [{"label": "background", "pixel_value": bg_class}]
        for i in range(n_classes):
            if i != bg_class:
                class_defs.append({"label": f"class{i}", "pixel_value": i})
        return ClassMapping(class_defs)

    def test_metrics(self):
        all_metrics_list = [metric for metric in EvalMetric]
        split = SplitType.VAL
        # test 1, should yield 1.0 for all metrics
        n_classes = 8
        cm = self.get_class_mapping(n_classes)
        pred_1 = torch.Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
        mask_1 = torch.Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
        histograms_1 = compute_histograms(pred_1, mask_1, n_classes)
        metrics = compute_metrics(histograms_1, all_metrics_list)
        for metric_key, metric_value in metrics.items():
            self.assertTrue(torch.equal(metric_value, Tensor([1.0 for _ in range(n_classes)])))
        metrics_tracker = MetricsTracker(all_metrics_list, split, "test", cm)
        metrics_tracker.update_metrics_from_image("img_1", pred_1, mask_1)
        metrics_micro = metrics_tracker.compute_total_metrics_micro_avg()
        metrics_macro = metrics_tracker.compute_total_metrics_macro_avg()
        for metric_key, metric_value in metrics_micro.items():
            self.assertTrue(metric_value == 1)
        for metric_key, metric_value in metrics_macro.items():
            self.assertTrue(metric_value == 1)

        # test 2, should yield 0.0 for all metrics
        n_classes = 8
        cm = self.get_class_mapping(n_classes)
        pred_2 = torch.Tensor(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
        mask_2 = torch.Tensor(np.array([[7, 6, 5, 4], [3, 2, 1, 0]]))
        histograms_2 = compute_histograms(pred_2, mask_2, n_classes)
        metrics = compute_metrics(histograms_2, all_metrics_list)
        for metric_key, metric_value in metrics.items():
            self.assertTrue(torch.equal(metric_value, Tensor([0.0 for _ in range(n_classes)])))
        metrics_tracker = MetricsTracker(all_metrics_list, split, "test", cm)
        metrics_tracker.update_metrics_from_image("img_1", pred_2, mask_2)
        metrics_micro = metrics_tracker.compute_total_metrics_micro_avg()
        metrics_macro = metrics_tracker.compute_total_metrics_macro_avg()
        for metric_key, metric_value in metrics_micro.items():
            self.assertTrue(metric_value == 0)
        for metric_key, metric_value in metrics_macro.items():
            self.assertTrue(metric_value == 0)

        # test 3, variety of metric values
        n_classes = 3
        pred_3 = torch.Tensor(np.array([[0, 1, 1, 1], [2, 1, 2, 1]]))
        mask_3 = torch.Tensor(np.array([[0, 1, 2, 1], [1, 1, 1, 0]]))
        tp_3 = Tensor([1, 3, 0])
        fp_3 = Tensor([0, 2, 2])
        fn_3 = Tensor([1, 2, 1])
        self._test_single_metrics_example(pred_3, mask_3, tp_3, fp_3, fn_3, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=0)
        self._test_single_metrics_example(pred_3, mask_3, tp_3, fp_3, fn_3, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=1)
        self._test_single_metrics_example(pred_3, mask_3, tp_3, fp_3, fn_3, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=2)
        self._test_single_metrics_example(pred_3, mask_3, tp_3, fp_3, fn_3, n_classes, all_metrics_list,
                                          ignore_background=False, background_class=0)

        # test 4, another variety of metric values
        n_classes = 4
        pred_4 = torch.Tensor(np.array([[0, 1, 1, 3, 2], [2, 3, 2, 1, 3]]))
        mask_4 = torch.Tensor(np.array([[0, 1, 2, 1, 2], [0, 3, 1, 0, 3]]))
        tp_4 = Tensor([1, 1, 1, 2])
        fp_4 = Tensor([0, 2, 2, 1])
        fn_4 = Tensor([2, 2, 1, 0])
        self._test_single_metrics_example(pred_4, mask_4, tp_4, fp_4, fn_4, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=0)
        self._test_single_metrics_example(pred_4, mask_4, tp_4, fp_4, fn_4, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=1)
        self._test_single_metrics_example(pred_4, mask_4, tp_4, fp_4, fn_4, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=2)
        self._test_single_metrics_example(pred_4, mask_4, tp_4, fp_4, fn_4, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=3)

        # test 5, another variety of metric values
        n_classes = 3
        pred_5 = torch.Tensor(np.array([[0, 1, 1, 1], [1, 0, 2, 1]]))
        mask_5 = torch.Tensor(np.array([[0, 2, 2, 2], [1, 0, 0, 0]]))
        tp_5 = Tensor([2, 1, 0])
        fp_5 = Tensor([0, 4, 1])
        fn_5 = Tensor([2, 0, 3])
        self._test_single_metrics_example(pred_5, mask_5, tp_5, fp_5, fn_5, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=0)
        self._test_single_metrics_example(pred_5, mask_5, tp_5, fp_5, fn_5, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=1)
        self._test_single_metrics_example(pred_5, mask_5, tp_5, fp_5, fn_5, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=2)
        self._test_single_metrics_example(pred_5, mask_5, tp_5, fp_5, fn_5, n_classes, all_metrics_list,
                                          ignore_background=False, background_class=0)

        # test 6, use values from test 3 and 5 for a batch test
        n_classes = 3
        pred_batch = torch.stack([pred_3, pred_5], dim=0)
        mask_batch = torch.stack([mask_3, mask_5], dim=0)
        tp_batch = tp_3 + tp_5
        fp_batch = fp_3 + fp_5
        fn_batch = fn_3 + fn_5
        self._test_single_metrics_example(pred_batch, mask_batch, tp_batch, fp_batch, fn_batch, n_classes,
                                          all_metrics_list, ignore_background=True, background_class=0)

        # test 7, binary case
        n_classes = 2
        pred_7 = torch.Tensor(np.array([[0, 1, 1, 1], [1, 1, 0, 0]]))
        mask_7 = torch.Tensor(np.array([[0, 1, 0, 1], [1, 1, 1, 0]]))
        tp_7 = Tensor([2, 4])
        fp_7 = Tensor([1, 1])
        fn_7 = Tensor([1, 1])
        self._test_single_metrics_example(pred_7, mask_7, tp_7, fp_7, fn_7, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=0)
        self._test_single_metrics_example(pred_7, mask_7, tp_7, fp_7, fn_7, n_classes, all_metrics_list,
                                          ignore_background=True, background_class=1)
        self._test_single_metrics_example(pred_7, mask_7, tp_7, fp_7, fn_7, n_classes, all_metrics_list,
                                          ignore_background=False, background_class=0)

    def _test_single_metrics_example(self, pred, mask, tp, fp, fn, n_classes, all_metrics_list, ignore_background=True,
                                     background_class=0):
        split = SplitType.VAL
        histograms = compute_histograms(pred, mask, n_classes)
        metrics = compute_metrics(histograms, all_metrics_list)

        target_metrics = self._calculate_per_class_target_metrics(tp, fp, fn)

        # test per class metrics
        for metric_key, metric_value in metrics.items():
            self.assertTrue(torch.equal(metric_value, target_metrics[metric_key]),
                            msg=f"metric {metric_key} is calculated to be {metric_value} but should be {target_metrics[metric_key]}")

        # test micro and macro averaged total metrics
        cm = self.get_class_mapping(n_classes, background_class)
        metrics_tracker = MetricsTracker(all_metrics_list, split, "test", cm, ignore_background=ignore_background)
        mapped_pred = cm.apply_class_mapping(pred)
        mapped_mask = cm.apply_class_mapping(mask)
        if len(pred.shape) == 2:
            metrics_tracker.update_metrics_from_image("img", mapped_pred, mapped_mask)

        elif len(pred.shape) == 3:
            metrics_tracker.update_metrics_from_batch(["img_1", "img_2"], mapped_pred, mapped_mask)
        else:
            raise ValueError("Wrong shape of pred tensors")

        metrics_micro = metrics_tracker.compute_total_metrics_micro_avg()
        metrics_macro = metrics_tracker.compute_total_metrics_macro_avg()

        # compute micro averaged target metrics
        # build sums without background class
        if ignore_background:
            tp_sum = torch.cat((tp[:background_class], tp[background_class + 1:])).sum().item()
            fp_sum = torch.cat((fp[:background_class], fp[background_class + 1:])).sum().item()
            fn_sum = torch.cat((fn[:background_class], fn[background_class + 1:])).sum().item()
        else:
            tp_sum = tp.sum().item()
            fp_sum = fp.sum().item()
            fn_sum = fn.sum().item()

        target_micro = self._calculate_per_class_target_metrics(tp_sum, fp_sum, fn_sum)

        # compute macro averaged target metrics
        target_macro = {}
        for metric_key, metric_value in target_metrics.items():
            if ignore_background:
                # get mean without background class
                target_macro[metric_key] = torch.mean(torch.cat((metric_value[:background_class],
                                                                 metric_value[background_class + 1:]))).item()
            else:
                target_macro[metric_key] = torch.mean(metric_value).item()

        # check micro and macro averaged metrics
        for metric_key, metric_value in metrics_micro.items():
            self.assertAlmostEqual(metric_value, target_micro[metric_key], places=6)
        for metric_key, metric_value in metrics_macro.items():
            self.assertAlmostEqual(metric_value, target_macro[metric_key], places=6)

        # compare with sklearn metrics
        macro_mode = "macro"
        micro_mode = "micro"
        ignore_class = background_class if ignore_background else None
        self._check_with_sklearn(all_metrics_list, target_macro, pred, mask, n_classes, macro_mode, ignore_class)
        self._check_with_sklearn(all_metrics_list, target_micro, pred, mask, n_classes, micro_mode, ignore_class)

    def _check_with_sklearn(self, all_metrics_list, target_metrics, pred, mask, n_classes, average_mode,
                            ignore_class=None):
        true_labels = mask.flatten().numpy().astype(int)
        pred_labels = pred.flatten().numpy().astype(int)

        # Remove the ignored class from the labels list
        labels = list(range(n_classes))
        if ignore_class is not None:
            labels.remove(ignore_class)

        for idx, metric_key in enumerate(all_metrics_list):
            if metric_key == EvalMetric.F1_SCORE:
                sklearn_metric = f1_score(true_labels, pred_labels, average=average_mode, labels=labels)
            elif metric_key == EvalMetric.IOU:
                sklearn_metric = jaccard_score(true_labels, pred_labels, average=average_mode, labels=labels)
            elif metric_key == EvalMetric.PRECISION:
                sklearn_metric = precision_score(true_labels, pred_labels, average=average_mode, labels=labels)
            elif metric_key == EvalMetric.RECALL:
                sklearn_metric = recall_score(true_labels, pred_labels, average=average_mode, labels=labels)
            else:
                continue

            self.assertAlmostEqual(sklearn_metric, target_metrics[metric_key], places=6,
                                   msg=f"Metric {metric_key} does not match with sklearn")

    def _calculate_per_class_target_metrics(self, tp, fp, fn):
        target_metrics = {}
        target_metrics[EvalMetric.DICE] = 2 * tp / (2 * tp + fp + fn)
        target_metrics[EvalMetric.PRECISION] = tp / (tp + fp)
        target_metrics[EvalMetric.RECALL] = tp / (tp + fn)
        target_metrics[EvalMetric.F1_SCORE] = 2 * target_metrics[EvalMetric.PRECISION] * target_metrics[
            EvalMetric.RECALL] / (target_metrics[EvalMetric.PRECISION] + target_metrics[EvalMetric.RECALL])
        target_metrics[EvalMetric.IOU] = tp / (tp + fp + fn)
        # replace nan
        for metric_key, metric_value in target_metrics.items():
            if isinstance(target_metrics[metric_key], Tensor):
                target_metrics[metric_key][torch.isnan(target_metrics[metric_key])] = 0
        return target_metrics
