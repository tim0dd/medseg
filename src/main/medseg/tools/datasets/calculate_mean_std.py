import click
import numpy as np
from PIL.Image import Image
from torch import Tensor

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager
from medseg.util.img_ops import calculate_mean_std, numpy_to_tensor, pil_to_tensor


@click.command()
@click.option('--path', type=str, required=True, help='Path to a minimal config.')
@click.option('--enable_transforms', '--t', is_flag=True,
              help='Apply the transform pipeline in the config before calculating.')
def calculate_mean_std_for_dataset(path: str, enable_transforms: bool):
    """Compute class imbalance for a dataset.

    Args:
        path (str): Path to a minimal configuration file.
        enable_transforms (bool): Whether to apply the transform pipeline in the config before calculating.
    """
    cfg = load_and_parse_config(path)
    dataset_manager = DatasetManager(cfg)
    ds = dataset_manager.get_train_dataset()

    def get_image_func(i: int) -> Tensor:
        if enable_transforms:
            img, _, _ = ds.__getitem__(i)
        else:
            img = ds.load_img(i)
        if isinstance(img, Tensor):
            return img
        elif isinstance(img, np.ndarray):
            return numpy_to_tensor(img)
        elif isinstance(img, Image):
            return pil_to_tensor(img)

    mean, std = calculate_mean_std(get_image_func, len(ds), ds.img_channels)
    for k, v in ds.transforms_manager.transform_times.items():
        avg_time = np.mean(v)
        print(f"{k}: {avg_time}")
    print(f"Mean and standard deviation for dataset {ds.__class__.__name__}")
    print(f"Transforms enabled: {enable_transforms}")
    print(f"Path to config: {path}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == '__main__':
    calculate_mean_std_for_dataset()
