from typing import Tuple, List

import click
import numpy as np
import torch
from tabulate import tabulate

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager
from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.util.path_builder import PathBuilder


@click.command()
@click.option('--config', '-c', type=str, required=True, help='Path to a minimal config.')
@click.option('--enable_transforms', '-t', is_flag=True,
              help='Apply the transform pipeline in the config before calculating.')
def compute_class_imbalance(config: str, enable_transforms: bool):
    """Compute class imbalance for a dataset.

    Args:
        config (str): Path to a minimal configuration file.
        enable_transforms (bool): Whether to apply the transform pipeline in the config before calculating.
    """
    cfg = load_and_parse_config(config)
    ds_manager = DatasetManager(cfg)
    ds_train = ds_manager.get_train_dataset()
    ds_val = ds_manager.get_val_dataset()
    ds_test = ds_manager.get_test_dataset()
    ds_all = DatasetManager.build_dataset(cfg, SplitType.ALL, include_transforms_manager=enable_transforms)
    pixel_count_train = count_pixels(ds_train, enable_transforms)
    pixel_count_val = count_pixels(ds_val, enable_transforms)
    pixel_count_test = count_pixels(ds_test, enable_transforms)
    pixel_count_all = count_pixels(ds_all, enable_transforms)
    class_defs = ds_train.get_class_defs()

    print("\nClass statistics:")
    table_data = []
    for class_def in class_defs:
        pixel_value = class_def['pixel_value']
        mean_train, std_train, pct_train = get_mean_std_pct_for_pixel_value(pixel_value, pixel_count_train)
        mean_val, std_val, percentage_val = get_mean_std_pct_for_pixel_value(pixel_value, pixel_count_val)
        mean_test, std_test, percentage_test = get_mean_std_pct_for_pixel_value(pixel_value, pixel_count_test)
        mean_all, std_all, percentage_all = get_mean_std_pct_for_pixel_value(pixel_value, pixel_count_all)
        row_data = [
            f"{class_def['label']} (pixel value: {pixel_value})",
            f"{mean_train:.2f}, {std_train:.2f}, {pct_train:.2f}%",
            f"{mean_val:.2f}, {std_val:.2f}, {percentage_val:.2f}%",
            f"{mean_test:.2f}, {std_test:.2f}, {percentage_test:.2f}%",
            f"{mean_all:.2f}, {std_all:.2f}, {percentage_all:.2f}%"
        ]
        table_data.append(row_data)

    headers = ["Class", "Train (Mean, Std, %)", "Validation (Mean, Std, %)", "Test (Mean, Std, %)",
               "All (Mean, Std, %)"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    non_predefined_pixel_counts = {
        "train": get_non_predefined_class_pixel_count(class_defs, pixel_count_train),
        "val": get_non_predefined_class_pixel_count(class_defs, pixel_count_val),
        "test": get_non_predefined_class_pixel_count(class_defs, pixel_count_test),
        "all": get_non_predefined_class_pixel_count(class_defs, pixel_count_all)
    }

    total_pixel_counts = {
        "train": sum([sum(img_data.values()) for img_data in pixel_count_train]),
        "val": sum([sum(img_data.values()) for img_data in pixel_count_val]),
        "test": sum([sum(img_data.values()) for img_data in pixel_count_test]),
        "all": sum([sum(img_data.values()) for img_data in pixel_count_all])
    }

    print("\n'Rogue' pixel values not belonging to any predefined class:")

    unique_rogue_pixel_values = set()
    for rogue_pixel_values in non_predefined_pixel_counts.values():
        unique_rogue_pixel_values.update(rogue_pixel_values.keys())

    table_data = []
    for pixel_value in sorted(unique_rogue_pixel_values):
        row_data = [f"Pixel value {pixel_value}"]
        for dataset_name in ["train", "val", "test", "all"]:
            count = non_predefined_pixel_counts[dataset_name].get(pixel_value, 0)
            percentage = 100 * count / total_pixel_counts[dataset_name]
            row_data.append(f"{count} ({percentage:.4f}%)")
        table_data.append(row_data)

    headers = ["", "Train", "Validation", "Test", "All"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    save_path = PathBuilder.out_builder().add(f"pixel_distribution_{ds_train.get_name()}.png").build()
    # plot_pixel_distribution(total_pixel_counts, class_defs, save_path=save_path)


def get_mean_std_pct_for_pixel_value(pixel_value: int, img_pixel_count_data: List[dict]) -> Tuple[
    float, float, float]:
    """Calculate mean, standard deviation, and percentage for a given class across all images.

    Args:
        pixel_value (int): Integer value of the pixel for the class.
        img        img_pixel_count_data (List[dict]): List of dictionaries containing pixel counts for each class.

    Returns:
        Tuple[float, float, float]: Mean, standard deviation, and percentage for the class.
    """
    pixel_counts = [img_data.get(pixel_value, 0) for img_data in img_pixel_count_data]
    total_pixel_counts = [sum(img_data.values()) for img_data in img_pixel_count_data]
    mean = np.mean(pixel_counts)
    std = np.std(pixel_counts)
    percentages = [100 * count / total_count for count, total_count in zip(pixel_counts, total_pixel_counts)]
    percentage_mean = np.mean(percentages)
    return float(mean), float(std), float(percentage_mean)


def count_pixels(dataset: MedsegDataset, enable_transforms: bool) -> List[dict]:
    """Count pixels for each class in the dataset.

    Args:
        dataset (MedsegDataset): MedsegDataset instance containing the dataset.
        enable_transforms (bool): Whether to apply the transform pipeline in the config before counting pixels.

    Returns:
        List[dict]: List of dictionaries containing pixel counts for each class.
    """

    img_pixel_count_data = [{} for _ in range(len(dataset))]
    for i in range(len(dataset)):
        img, mask, real_i = dataset.load_img_mask(i)
        if enable_transforms:
            _, mask = dataset.transforms_manager.apply_transforms(img, mask, real_i)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        for pixel_value in torch.unique(mask):
            class_mask = mask == pixel_value
            pixel_count = torch.sum(class_mask).item()
            img_pixel_count_data[i][pixel_value.item()] = pixel_count

    return img_pixel_count_data


def get_non_predefined_class_pixel_count(class_defs, img_pixel_count_data: List[dict]) -> dict:
    """Calculate the pixel count of non-predefined classes in the dataset.

    Args:
        class_defs (List[dict]): List of dictionaries containing class definitions.
        img_pixel_count_data (List[dict]): List of dictionaries containing pixel counts for each class.

    Returns:
        dict: Dictionary containing rogue pixel values and their counts.
    """
    predefined_pixel_values = {class_def['pixel_value'] for class_def in class_defs}
    non_predefined_pixel_count = {}

    for img_data in img_pixel_count_data:
        for pixel_value, count in img_data.items():
            if pixel_value not in predefined_pixel_values:
                non_predefined_pixel_count[pixel_value] = non_predefined_pixel_count.get(pixel_value, 0) + count

    return non_predefined_pixel_count


# def plot_pixel_distribution(img_pixel_counts: Dict[List[Dict]], class_defs: List[dict], save_path: str) -> None:
#     """Plot the pixel distribution data for train, validation, and test datasets using seaborn.
#
#     Args:
#         img_pixel_count_data_train (List[dict]): Pixel count data for the training dataset.
#         img_pixel_count_data_val (List[dict]): Pixel count data for the validation dataset.
#         img_pixel_count_data_test (List[dict]): Pixel count data for the test dataset.
#         class_defs (List[dict]): List of dictionaries containing class definitions.
#     """
#
#     data = []
#     for dataset_name, pixel_count_list in img_pixel_counts.items():
#         for img_data in pixel_count_list:
#             for class_def in class_defs:
#                 pixel_value = class_def['pixel_value']
#                 label = class_def['label']
#                 count = img_data.get(pixel_value, 0)
#                 data.append({"Dataset": dataset_name, "Class": label, "Count": count})
#
#     df = pd.DataFrame(data)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="Class", y="Count", hue="Dataset", data=df)
#     plt.title("Pixel Distribution per Class")
#     # save
#     plt.savefig(save_path)


if __name__ == '__main__':
    compute_class_imbalance()
