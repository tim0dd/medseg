import glob
import os
import platform
import sys
from logging import Logger
from typing import Callable, Optional, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from beartype import beartype
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists, getPrediction
from cityscapesscripts.helpers.csHelpers import colors, printError
from tensorboardX import SummaryWriter
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from medseg.data.dataset_manager import DatasetManager
from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.evaluation.loss_tracker import LossTracker
from medseg.evaluation.metrics_manager import MetricsManager
from medseg.evaluation.segmentation_visualizer import SegmentationVisualizer, ImageSaveMode
from medseg.models.model_builder import build_model
from medseg.training.loss.loss_builder import get_loss_module
from medseg.training.trainer_state import TrainerState
from medseg.util.date_time import get_current_date_time_str
from medseg.util.img_ops import logits_to_segmentation_mask
from medseg.util.logger import create_custom_logger
from medseg.util.path_builder import PathBuilder


class Evaluator:
    """
    Class for evaluating a model on a given dataset split.
    """

    @beartype
    def __init__(self, cfg: dict, model: nn.Module, compiled_model: Callable, split: SplitType,
                 dataset_manager: DatasetManager, metrics_manager: MetricsManager, device: torch.device,
                 loss_func: Callable, logger: Logger, base_pb: PathBuilder, save_sample_segmentations: bool = False,
                 eval_object_sizes: bool = False):
        self.cfg = cfg
        self.model = model
        self.compiled_model = compiled_model
        self.split = split
        self.dataset_manager = dataset_manager
        self.metrics_manager = metrics_manager
        self.device = device
        self.loss_func = loss_func
        self.loss_tracker = LossTracker(self.split, self.metrics_manager.tbx_writer)
        self.logger = logger
        self.save_sample_segmentations = save_sample_segmentations
        self.save_modes = [ImageSaveMode.RANDOM_SUBSET, ImageSaveMode.WORST]
        self.base_pb = base_pb
        self.eval_object_sizes = eval_object_sizes
        self.metrics_manager.set_eval_object_sizes(eval_object_sizes, self.split)

    @classmethod
    @beartype
    def from_trainer_state(cls, state: TrainerState, split: SplitType, save_sample_segmentations: bool = False,
                           eval_object_sizes: bool = False):
        return cls(state.cfg, state.model, state.compiled_model, split, state.dataset_manager, state.metrics_manager,
                   state.device, state.loss_func, state.logger, PathBuilder.trial_out_builder(state.cfg),
                   save_sample_segmentations, eval_object_sizes)

    @classmethod
    @beartype
    def from_checkpoint(cls, checkpoint: dict, path: str, split: SplitType):
        path = os.path.dirname(path)  # get path without filename
        base_pb = PathBuilder().add(path)
        cfg = checkpoint['cfg']
        if 'trial_name' not in cfg:
            trial_name = f"{cfg['architecture']['model_name']}_{get_current_date_time_str()}_EVAL"
            cfg['trial_name'] = trial_name
        else:
            trial_name = cfg['trial_name']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_manager = DatasetManager(cfg)
        class_mapping = None
        for dataset in dataset_manager.datasets.values():
            class_mapping = dataset.class_mapping if class_mapping is None else class_mapping
        assert class_mapping is not None, "No class mapping found in dataset manager"
        out_channels = class_mapping.num_classes if class_mapping.multiclass else 1
        model = build_model(cfg, out_channels=out_channels)
        model.to(device)
        model.load_state_dict(checkpoint['model'])
        # do not compile model here in any case, as the compilation would take far too long relative to the eval
        compiled_model = model
        log_path = base_pb.clone().add('eval_log.txt').build()
        logger = create_custom_logger(f"logger_{cfg['trial_name']}", log_path)

        tbx_writer = SummaryWriter(log_dir=base_pb.clone().build())
        metrics_manager = MetricsManager(cfg, class_mapping, trial_name, tbx_writer, base_pb=base_pb)
        multiclass = class_mapping.multiclass
        loss_from_cfg = get_loss_module(cfg)
        loss_func = model.default_loss_func(multiclass) if loss_from_cfg is None else loss_from_cfg
        save_sample_segmentations = cfg['settings'].get('save_sample_segmentations', False)
        eval_object_sizes = cfg['settings'].get('eval_object_sizes', False)
        return cls(cfg, model, compiled_model, split, dataset_manager, metrics_manager, device, loss_func, logger,
                   base_pb, save_sample_segmentations, eval_object_sizes)

    @beartype
    def evaluate(self, training_epoch: int, for_epochs: Optional[int] = None, is_final_eval: bool = False) -> Dict[str,
        MetricsManager]:
        """
        Evaluates the model on the calculates the evaluation metrics, and syncs the results to TensorBoard.

        Args:
            training_epoch (int): The current epoch of the training run
            for_epochs (Optional[int]): The number of epochs to evaluate for. If None, evaluation is run once
            is_final_eval (bool): Whether this is the final evaluation of the model.
        """
        metrics_managers = dict()
        if self.dataset_manager.has_split(self.split):
            dataset, loader = self.dataset_manager.get_dataset_and_loader(self.split)
            mm = self.evaluate_dataset(dataset, loader, training_epoch, False, for_epochs, is_final_eval=is_final_eval)
            metrics_managers[dataset.get_name().lower()] = mm

        if self.split == SplitType.TEST and self.dataset_manager.has_aux_test_datasets():
            for dataset, loader in self.dataset_manager.get_aux_test_datasets_and_loaders():
                mm = self.evaluate_dataset(dataset, loader, training_epoch, True, for_epochs,
                                           is_final_eval=is_final_eval)
                mm.save_full_metrics()
                metrics_managers[dataset.get_name().lower()] = mm
        return metrics_managers

    @beartype
    def evaluate_dataset(self,
                         dataset: MedsegDataset,
                         loader: DataLoader,
                         training_epoch: int,
                         is_aux_test_data: bool = False,
                         for_epochs: Optional[int] = None,
                         predictions_hook: Optional[Callable[[str, List[str], Tensor], None]] = None,
                         is_final_eval: bool = False
                         ) -> MetricsManager:
        """
        Evaluates the model on the specified dataset split, calculates the evaluation metrics, and syncs the results
        to TensorBoard.

        Args:
            dataset (MedsegDataset): The dataset to evaluate on.
            loader (DataLoader): The DataLoader for the dataset.
            training_epoch (int): The current epoch of the training run
            is_aux_test_data (bool): Whether the dataset is an auxiliary test dataset.
            for_epochs (Optional[int]): The number of epochs to evaluate for. If None, evaluation is run once
            predictions_hook (Optional[Callable[[str, List[str], Tensor], None]]): A hook to call after each batch
            is_final_eval (bool): Whether this is the final evaluation of the model.

        Returns:
            metrics_tracker: The MetricsTracker instance containing the computed evaluation metrics.
        """

        self.model.eval()
        self.loss_tracker.reset()

        metrics_manager = self._get_metrics_manager(dataset, not is_aux_test_data)
        metrics_tracker = metrics_manager.add_tracker(self.split)
        dataset_prefix = dataset.get_name() if is_aux_test_data else None
        eval_epochs = for_epochs if for_epochs is not None else 1
        self.logger.info(f"Evaluating {dataset.get_name()}'s {self.split.name} set for {eval_epochs} epoch(s)...")
        amend_img_filenames = eval_epochs > 1
        perform_cityscapes_eval = dataset.get_name().lower() == 'cityscapes'
        cityscapes_eval_pb = self.base_pb.clone().add('cityscapes_eval')
        slide_inference = perform_cityscapes_eval

        for current_eval_epoch in range(1, eval_epochs + 1):
            cityscapes_eval_epoch_pb = cityscapes_eval_pb.clone().add(f'epoch_{current_eval_epoch}')
            with torch.no_grad():
                for ([images, masks], ids) in loader:
                    images = images.to(device=self.device, dtype=torch.float)
                    masks = masks.to(device=self.device, dtype=torch.long if dataset.is_multiclass() else torch.float)
                    if slide_inference:
                        # TODO: generalize and fetch stride and crop size from config
                        stride = (768, 768)
                        crop_size = (1024, 1024)
                        num_classes = dataset.class_mapping.num_classes
                        predictions = self.slide_inference(images, stride, crop_size, num_classes)
                    else:
                        predictions = self.compiled_model(images)
                    masks = masks.squeeze(1) if dataset.is_multiclass() else masks
                    loss = self.loss_func(predictions, masks)
                    self.loss_tracker.update(loss)
                    predictions = logits_to_segmentation_mask(
                        predictions).int() if dataset.is_multiclass() else predictions > 0.5
                    img_filenames = [dataset.get_image_file_name(real_i) for real_i in ids]

                    if perform_cityscapes_eval:
                        to_regular_id_mapping = dataset.get_train_id_to_regular_ids_mapping()
                        pixel_to_regular_id_tensor = torch.full((256,), 0)
                        for pixel_id, regular_id in to_regular_id_mapping.items():
                            pixel_to_regular_id_tensor[pixel_id] = regular_id
                        assert to_regular_id_mapping is not None
                        for i in range(predictions.shape[0]):
                            pred = predictions[i].clone().cpu().int()
                            pred = dataset.class_mapping.revert_class_mapping(pred)
                            pred = pixel_to_regular_id_tensor[pred]
                            pred = pred.view(pred.shape[0], -1)
                            pred = Image.fromarray(pred.numpy().astype(np.uint8), "L")
                            pred.save(cityscapes_eval_epoch_pb.clone().add(img_filenames[i]).build())

                    if amend_img_filenames:
                        # if we are evaluating for multiple epochs, we need to amend the image filenames, otherwise
                        # metrics dicts for the respective img id will be overwritten every time
                        img_filenames = [f"{img_filename}_epoch_{current_eval_epoch}" for img_filename in img_filenames]
                    metrics_tracker.update_metrics_from_batch(img_filenames, predictions.cpu(), masks.cpu())
                    if predictions_hook is not None:
                        predictions_hook(dataset.get_name(), img_filenames, predictions.clone().cpu().int())

                    #
                    # if self.save_sample_segmentations:
                    #     img_folder = f"images_{self.split.value}"
                    #     img_save_pb = self.base_pb.clone().add(img_folder)
                    #     img_size = self.cfg['architecture'].get('in_size', 512)
                    #     visualizer = SegmentationVisualizer(self.model, self.device, dataset, img_save_pb, img_size)
                    #     for i in range(predictions.shape[0]):
                    #         visualizer.save_segmentation_sample(img_filenames[i], '', (images[i].clone() * 255),
                    #                                             predictions[i].clone(), masks[i].clone(),
                    #                                             revert_mask_class_mapping=True)

        if perform_cityscapes_eval:
            ds_path = dataset.dataset_path
            self.evaluate_cityscapes(ds_path, cityscapes_eval_pb.clone().build())
        metrics_tracker.compute_total_metrics()
        metrics_tracker.sync_to_tensorboard(training_epoch, ds_prefix=dataset_prefix)
        self.logger.info(f"Metrics for {self.split.name} set:")
        self.logger.info(metrics_tracker.get_metrics_summary())
        mean_loss = self.loss_tracker.compute_mean()
        self.loss_tracker.sync_to_tensorboard()
        metrics_manager.add_mean_loss(mean_loss, self.split, training_epoch)

        if self.save_sample_segmentations and eval_epochs <= 1 and is_final_eval:
            if for_epochs is not None and for_epochs > 1:
                self.logger.info(f"Evaluation is running in multi-epoch mode, so sample segmentations will only be "
                                 f"saved for the first epoch.")
            self.logger.info(f"Saving sample segmentations {dataset.get_name()}'s {self.split.name} set...")
            img_folder = f"images_{dataset.get_name().lower()}_{self.split.name}"
            img_save_pb = self.base_pb.clone().add(img_folder)
            img_size = self.cfg['architecture'].get('in_size', 512)
            # TODO: adapt to slide inference
            visualizer = SegmentationVisualizer(self.model, self.device, dataset, img_save_pb, img_size)
            worst_ids = None
            if ImageSaveMode.WORST in self.save_modes:
                worst_ids = metrics_manager.get_bottom_k_img_ids(self.split, k=50)
            n_random_samples = 100 if ImageSaveMode.RANDOM_SUBSET in self.save_modes else None
            visualizer.save_segmentations(worst_ids, n_random_samples)
            del visualizer
        return metrics_manager

    def slide_inference(self, img: Tensor, stride: Tuple[int, int],
                        crop_size: Tuple[int, int], num_classes: int, rescale: bool = False,
                        orig_shape: Optional[Tuple[int, int]] = None):
        """
        Inference by sliding-window with overlap.
        Adapted from https://github.com/open-mmlab/mmsegmentation
        Args:
            img: input image(s)
            stride: stride of sliding window
            crop_size: crop size of sliding window
            num_classes: number of classes
            rescale: whether to rescale the output to original size
            orig_shape: original shape of the image, if rescale is True, orig_shape must be specified
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.size()

        # divide into grids
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # initialize zero-tensors
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # calculate the top-left and bottom-right coordinates of the crop
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                # crop image accordingly
                crop_img = img[:, :, y1:y2, x1:x2]
                # perform prediction
                crop_seg_logit = self.compiled_model(crop_img)
                # add to total prediction, add padding to fit size
                preds += F.pad(crop_seg_logit,
                               [int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)])
                # track the area of the image that is already done
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        if rescale:
            preds = F.interpolate(
                preds,
                size=orig_shape,
                mode='bilinear',
                align_corners=False)
        return preds

    def evaluate_cityscapes(self, cityscapes_ds_path: str, cityscapes_preds_path: str):
        class CityArgs(object):
            def __init__(self):
                self.cityscapesPath = cityscapes_ds_path

                # subfolders of cityscapespath are epoch_1, epoch_2, etc.
                # collect all files ending with "_gtFine_labelIds.png" within them
                self.groundTruthSearch = os.path.join(cityscapes_ds_path, "masks", "*_gtFine_labelIds.png")

                # Remaining params
                self.evalInstLevelScore = True
                self.evalPixelAccuracy = False
                self.evalLabels = []
                self.printRow = 5
                self.normalized = True
                self.colorized = hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and platform.system() == 'Linux'
                self.bold = colors.BOLD if self.colorized else ""
                self.nocol = colors.ENDC if self.colorized else ""
                self.JSONOutput = True
                self.exportFile = os.path.join(cityscapes_preds_path, "results.json")
                self.quiet = False

                self.avgClassSize = {
                    "bicycle": 4672.3249222261,
                    "caravan": 36771.8241758242,
                    "motorcycle": 6298.7200839748,
                    "rider": 3930.4788056518,
                    "bus": 35732.1511111111,
                    "train": 67583.7075812274,
                    "car": 12794.0202738185,
                    "person": 3462.4756337644,
                    "truck": 27855.1264367816,
                    "trailer": 16926.9763313609,
                }

                # store some parameters for finding predictions in the args variable
                # the values are filled when the method getPrediction is first called
                self.predictionPath = cityscapes_preds_path
                self.predictionWalk = None

        args = CityArgs()

        predictionImgList = []

        # use the ground truth search string specified above
        groundTruthImgList = glob.glob(args.groundTruthSearch)
        if not groundTruthImgList:
            printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
                args.groundTruthSearch))
        # get the corresponding prediction for each ground truth imag
        for gt in groundTruthImgList:
            predictionImgList.append(getPrediction(args, gt))
        # evaluate
        evaluateImgLists(predictionImgList, groundTruthImgList, args)

    @beartype
    def _get_metrics_manager(self, dataset: MedsegDataset, reuse_metrics_manager: bool = True) -> MetricsManager:
        if reuse_metrics_manager:
            return self.metrics_manager
        else:
            tbx_writer = self.metrics_manager.tbx_writer
            trial_name = self.metrics_manager.trial_name
            ds_prefix = dataset.get_name().lower()
            metrics_manager = MetricsManager(self.cfg,
                                             dataset.class_mapping,
                                             trial_name,
                                             tbx_writer,
                                             ds_prefix,
                                             base_pb=self.base_pb)
            metrics_manager.set_eval_object_sizes(self.eval_object_sizes, self.split)
            return metrics_manager
