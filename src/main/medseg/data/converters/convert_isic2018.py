import os

import click

from medseg.data.converters.converters import create_dataset_predefined_split
from medseg.data.converters.helpers import check_in_folder_paths


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_isic2018(in_path=None, out_path=None, copy_files=True):
    """
    Convert the original ISIC-2018 (Task 1) folder structure and add an index.csv and classes.csv file.
    Original structure is as follows:
    - isic2018



    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    train_img_path = os.path.join(in_path, "ISIC2018_Task1-2_Training_Input")
    train_mask_path = os.path.join(in_path, "ISIC2018_Task1_Training_GroundTruth")

    val_img_path = os.path.join(in_path, "ISIC2018_Task1-2_Validation_Input")
    val_mask_path = os.path.join(in_path, "ISIC2018_Task1_Validation_GroundTruth")

    test_img_path = os.path.join(in_path, "ISIC2018_Task1-2_Test_Input")
    test_mask_path = os.path.join(in_path, "ISIC2018_Task1_Test_GroundTruth")

    check_in_folder_paths(train_img_path, train_mask_path, val_img_path, val_mask_path, test_img_path, test_mask_path)

    train_paths = (train_img_path, train_mask_path)
    val_paths = (val_img_path, val_mask_path)
    test_paths = (test_img_path, test_mask_path)

    class_dict = {"background": 0, "lesion": 255}

    create_dataset_predefined_split([train_paths], [val_paths], [test_paths], out_path, class_dict,
                                    exclude_files=None, strict_filename_matching=False, copy_files=copy_files)


if __name__ == "__main__":
    convert_isic2018()
