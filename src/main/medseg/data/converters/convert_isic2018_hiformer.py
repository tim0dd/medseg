import glob
import os

import click

from medseg.data.converters.converters import create_dataset_predefined_split
from medseg.data.converters.helpers import check_in_folder_paths


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_isic2018_hiformer(in_path=None, out_path=None, copy_files=True):
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

    # to reproduce dataset ordering and splits in HiFormer, we use a similar method as in
    # https://github.com/rezazad68/BCDU-Net/blob/master/Skin%20Lesion%20Segmentation/Prepare_ISIC2018.py
    train_img_path = os.path.join(in_path, "ISIC2018_Task1-2_Training_Input")
    train_mask_path = os.path.join(in_path, "ISIC2018_Task1_Training_GroundTruth")
    check_in_folder_paths(train_img_path, train_mask_path)
    Dataset_add = 'dataset_isic18/'
    train_paths = (train_img_path, train_mask_path)
    img_files = glob.glob(os.path.join(train_img_path, "*.jpg"))
    indices = [i for i in range(2594)]
    train_set = indices[0:1815]
    val_set = indices[1815:1815 + 259]
    test_set = indices[1815 + 259:2594]
    a = b[0:len(Dataset_add)]
    b = b[len(b) - 16: len(b) - 4]


    class_dict = {"background": 0, "lesion": 255}

    create_dataset_predefined_split([train_paths], [val_paths], [test_paths], out_path, class_dict,
                                    exclude_files=None, strict_filename_matching=False, copy_files=copy_files)


if __name__ == "__main__":
    convert_isic2018_hiformer()
