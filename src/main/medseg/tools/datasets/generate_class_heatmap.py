from typing import Callable, Optional
from typing import List

import click
import matplotlib
import numpy as np
import seaborn as sns
from beartype import beartype
from matplotlib import pyplot as plt, gridspec
from matplotlib.figure import Figure
from torchvision.transforms import functional as F

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager
from medseg.data.split_type import SplitType
from medseg.util.path_builder import PathBuilder


@click.command()
@click.option('--cfg_path', '-c', 'cfg_paths', multiple=True, required=True, help='Path(s) to minimal configs.')
@click.option('--enable_transforms', '-t', is_flag=True,
              help='Apply the transform pipeline in the config before calculating.')
@click.option('--class_index', '-i', type=int, required=True, help='Index of the class to generate the heatmap for. '
                                                                   'This is the index that the class has after the '
                                                                   'mapping of pixel values to the indices (i.e. '
                                                                   'index according to classes.csv')
@click.option('--out_path', '-o', type=str, required=True, help='Path to save the heatmap to.')
@click.option('--split', '-s', type=click.Choice(['train', 'val', 'test', 'all'], case_sensitive=False), required=True,
              help='Which split to use: train, val, or test.')
@click.option('--label', '-l', 'labels', multiple=True, required=False,
              help='Labels for the plots in order of config paths.')
def generate_class_heatmaps(cfg_paths: List[str], class_index: int, enable_transforms: bool, out_path: str, split: str,
                            labels: List[str] = None):
    """Compute class heatmaps for a dataset.

    Args:
        cfg_paths (List[str]): Paths to minimal configuration files.
        class_index (int): Index of the class to generate the heatmap for.
        enable_transforms (bool): Whether to apply the transform pipeline in the config before calculating.
        out_path (str): Path to save the heatmaps to.
        split (str): Which split to use: train, val, or test.
    """
    matplotlib.use('Agg')
    class_count_arrays = []
    vmin, vmax = 0, 255

    # load first cfg
    first_cfg = load_and_parse_config(cfg_paths[0])

    def get_ds(cfg, split):
        ds_manager = DatasetManager(cfg)
        if split.lower() == 'all':
            return DatasetManager.build_dataset(cfg, SplitType.ALL, include_transforms_manager=True)
        elif split.lower() == 'train':
            return ds_manager.get_train_dataset()
        elif split.lower() == 'val':
            return ds_manager.get_val_dataset()
        elif split.lower() == 'test':
            return ds_manager.get_test_dataset()
        else:
            raise ValueError(f"Unknown split: {split}")

    ds_name = get_ds(first_cfg, split).get_name()

    for idx, cfg_path in enumerate(cfg_paths):
        cfg = load_and_parse_config(cfg_path)
        current_ds = get_ds(cfg, split)
        if ds_name != current_ds.get_name():
            print(f"Warning: Datasets do not match between the configs: {ds_name} vs {current_ds.get_name()}")

        def get_mask(i: int) -> np.ndarray:
            if enable_transforms:
                _, mask, _ = current_ds.__getitem__(i)
            else:
                mask = current_ds.load_mask(i)
                # use tensor conversion ensure that the expected transpose is applied
                F.pil_to_tensor(mask)
            return mask.numpy()

        class_count_array = calculate_class_count(class_index, get_mask, current_ds.indices)
        vmin = min(class_count_array.min(), vmin)
        vmax = max(class_count_array.max(), vmax)
        class_count_arrays.append(class_count_array)

    out_pb = PathBuilder().add(out_path)
    file_prefix = f"{ds_name.lower()}_heatmap_"
    labels = list(labels) if labels is not None else None
    heatmap_fig = generate_class_heatmap(class_count_arrays, int(vmin), int(vmax), labels=labels)
    heatmap_fig.savefig(out_pb.clone().add(f"{file_prefix}{split}.png").build(), bbox_inches='tight')


@beartype
def calculate_class_count(class_pixel: int, get_mask_func: Callable[[int], np.ndarray],
                          indices: List[int]) -> np.ndarray:
    class_count_array = None
    for i in range(len(indices)):
        mask = get_mask_func(i)
        mask = (mask == class_pixel).astype(int)
        class_count_array = mask if class_count_array is None else class_count_array + mask

    if len(class_count_array.shape) == 3 and class_count_array.shape[0] == 1:
        class_count_array = class_count_array[0]

    return class_count_array


@beartype
def generate_class_heatmap(class_count_arrays: List[np.ndarray], vmin: int, vmax: int,
                           labels: Optional[List[str]] = None) -> Figure:
    plt.rcParams.update({'font.size': 18})
    n = len(class_count_arrays)
    fig = plt.figure(figsize=(8 * n, 8))
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[1] * n + [0.05])

    for i, class_count_array in enumerate(class_count_arrays):
        ax = plt.subplot(gs[i])
        label = labels[i] if labels and i < len(labels) else ''
        if i == n - 1:
            # enable colorbar for last plot
            cbar_ax = fig.add_axes([0.95, ax.get_position().y0, 0.02, ax.get_position().height])
            cbar = sns.heatmap(class_count_array, cmap='Spectral_r', vmin=vmin, vmax=vmax, ax=ax, cbar=True,
                               cbar_ax=cbar_ax)
            cbar.collections[0].colorbar.set_label('Polyp class pixel count', labelpad=15)
        else:
            sns.heatmap(class_count_array, cmap='Spectral_r', vmin=vmin, vmax=vmax, ax=ax, cbar=False)

        ax.set_xticks([0, class_count_array.shape[1] - 1])
        ax.set_yticks([0, class_count_array.shape[0] - 1])
        ax.set_xticklabels([0, class_count_array.shape[1] - 1])
        ax.set_yticklabels([0, class_count_array.shape[0] - 1])
        ax.set_xlabel('Pixel x-coordinate')
        ax.set_ylabel('Pixel y-coordinate')
        ax.set_title(label, y=1.05)
        ax.set_aspect('equal')
    fig.tight_layout()
    return fig



if __name__ == '__main__':
    generate_class_heatmaps()
