import os

import click
from torch import Tensor
from torchvision.transforms import functional as F

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager
from medseg.evaluation.segmentation_visualizer import SegmentationVisualizer
from medseg.util.img_ops import tensor_to_pil
from medseg.util.path_builder import PathBuilder


@click.command()
@click.option('--cfg_path', '--c', type=str, required=True, help='Path to a minimal config.')
@click.option('--out', '--o', type=str, required=False, help='Output path.')
@click.option('--enable_transforms', '--t', is_flag=True,
              help='Apply the transform pipeline in the config before calculating.')
def draw_all_masks(cfg_path: str, out: str, enable_transforms=True):
    cfg = load_and_parse_config(cfg_path)
    dataset_manager = DatasetManager(cfg)
    out_pb = PathBuilder.out_builder().add("segs")
    for split, ds in dataset_manager.datasets.items():
        for i in range(len(ds)):
            img, mask, real_i = ds.__getitem__(i) if enable_transforms else ds.load_img_mask(i)
            if enable_transforms:
                # mask = ds.class_mapping.revert_class_mapping(mask)
                img = img * 255
            if isinstance(img, Tensor):
                img = tensor_to_pil(img)
            if isinstance(mask, Tensor):
                mask = F.to_pil_image(mask.float())
            gt_image = SegmentationVisualizer.draw_mask_overlay(img, mask, (0, 200, 128))
            file_name = ds.images[i]
            if out is not None:
                save_path = os.path.join(out, ds.get_name(), split.value, f"{file_name}_seg.png")
            else:
                save_path = out_pb.clone().add(ds.get_name()).add(split.value).add(f"{file_name}_seg.png").build()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gt_image.save(save_path)


if __name__ == '__main__':
    draw_all_masks()
