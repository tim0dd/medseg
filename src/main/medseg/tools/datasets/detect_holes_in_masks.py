import click
import cv2
import numpy as np

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager


@click.command()
@click.option('--config', '-c', type=str, required=True, help='Path to a minimal dataset config in order to load the '
                                                              'correct dataset.')
@click.option('--enable_transforms', '-t', type=bool, required=True,
              help='Apply the transform pipeline in the config before calculating.')
def detect_holes_in_masks(config: str, enable_transforms: bool):
    """
    Detect holes in segmentation masks. Loops over all masks, and prints a line for each detected hole in a mask.
    The line contains the pixel value of the mask, the class label, and the pixel size of the hole.
    """

    cfg = load_and_parse_config(config)
    ds_manager = DatasetManager(cfg)
    ds_train = ds_manager.get_train_dataset()
    ds_val = ds_manager.get_val_dataset()
    ds_test = ds_manager.get_test_dataset()
    class_defs = ds_train.get_class_defs()

    for split_name, dataset in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        print(f"\n{split_name} dataset:")
        for i in range(len(dataset)):
            _, mask, _ = dataset.__getitem__(i) if enable_transforms else dataset.load_img_mask(i)
            mask = np.array(mask, dtype=np.uint8)
            for pixel_value in np.unique(mask):
                # Check if the pixel value is in the class definitions
                class_def = next((d for d in class_defs if d["pixel_value"] == pixel_value), None)
                if class_def:
                    # Find contours of holes
                    _, contours, _ = cv2.findContours((mask == pixel_value).astype(np.uint8), cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)

                    # Loop over contours and calculate the area of each hole
                    for contour in contours:
                        hole_area = cv2.contourArea(contour)
                        print(f"{class_def['label']} (pixel value {pixel_value}): hole of {hole_area:.2f} pixels")


if __name__ == '__main__':
    detect_holes_in_masks()
