from typing import List

from medseg.evaluation.metrics import EvalMetric
from medseg.evaluation.metrics_tracker import MetricsTracker


class EarlyStop:
    """A class implementing an early stopping mechanism for model training.

    Attributes:
        metric (EvalMetric): The metric used to monitor model performance.
        tolerance (int): The number of epochs with no improvement before stopping.
        min_delta (float): The minimum change in the monitored metric to qualify as an improvement.
        counter (int): A counter for the number of epochs with no improvement.
        stop_triggered (bool): Whether early stopping has been triggered.
    """

    def __init__(self, metric=EvalMetric.IOU, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.metric = metric if isinstance(metric, EvalMetric) else EvalMetric(metric)
        self.counter = 0
        self.stop_triggered = False

    def check_metric(self, metrics_trackers: List[MetricsTracker]):
        """Checks if the metric has improved and updates the counter.

        Args:
            metrics_trackers (List[MetricsTracker]): A list of metrics trackers for previous epochs.
        """
        if metrics_trackers is None or len(metrics_trackers) < 2:
            return
        if self.metric not in metrics_trackers[-1].tracked_metrics \
                or self.metric not in metrics_trackers[-2].tracked_metrics:
            raise Warning(f"Early stop metric {self.metric} is not tracked in the given metrics tracker")

        metric_current = metrics_trackers[-1].total_metrics[self.metric]
        metric_previous = metrics_trackers[-2].total_metrics[self.metric]

        if (metric_current - metric_previous) < self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.stop_triggered = True
        else:
            self.counter = 0

    def state_dict(self):
        """Returns the state dictionary of the EarlyStop instance.

        Returns:
            dict: A dictionary containing the state of the EarlyStop instance.
        """
        return {
            'counter': self.counter,
            'stop_triggered': self.stop_triggered
        }

    def load_state_dict(self, state_dict: dict):
        """Initializes an EarlyStop instance from a state dictionary.

        Args:
            state_dict (dict): The state dictionary to initialize the EarlyStop instance.
        """
        self.counter = state_dict['counter']
        self.stop_triggered = state_dict['stop_triggered']


def get_early_stop(cfg: dict):
    """Returns an EarlyStop instance based on the provided configuration.

    Args:
        cfg (dict): The configuration dictionary containing early stopping settings.

    Returns:
        EarlyStop: An EarlyStop instance or None if early stopping is not enabled.
    """
    if cfg['early_stop'] is None or cfg['early_stop'] is False:
        return None
    elif cfg['early_stop'] is True:
        return EarlyStop()
    else:
        EarlyStop(**cfg['early_stop'])
