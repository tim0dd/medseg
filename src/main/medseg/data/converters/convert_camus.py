import os
import shutil
from typing import List

import click
from beartype import beartype
from medpy import io as medpy_io

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@beartype
def convert_camus(in_path: str, out_path: str):
    """
    Conversion script for the CAMUS dataset.

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved

    Returns:
    None
    """

    train_path = os.path.join(in_path, "training")
    test_path = os.path.join(in_path, "testing")
    check_in_folder_paths(train_path, test_path)

    train_folders = [f.path for f in os.scandir(train_path) if f.is_dir()]

    temp_train_path = os.path.join(out_path, "temp")
    os.makedirs(temp_train_path, exist_ok=True)
    # convert to images, save in temp folder
    convert_folders(temp_train_path, train_folders, is_test=False)

    img_paths_in = os.path.join(temp_train_path, "images")
    mask_paths_in = os.path.join(temp_train_path, "masks")
    folders_in = [(img_paths_in, mask_paths_in)]

    class_dict = {"background": 0, "left_ventricle": 85, "left_ventricle_wall ": 170, "left_atrium": 255}
    ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}

    create_dataset(folders_in, out_path, class_dict, ratios, seed=None, exclude_files=None, copy_files=True)
    shutil.rmtree(temp_train_path)


@beartype
def convert_folders(out_path: str, folder_paths: List[str], is_test: bool = False) -> List[str]:
    filenames = []
    file_ending = ".png"
    # special case for test set because patient numbers are labelled from 0001 to 0050 which would overlap
    # with training set labels... training set stops at 450, so the test set should start at 451
    train_count = 1
    test_count = 451
    save_path_img = os.path.join(out_path, "images")
    save_path_mask = os.path.join(out_path, "masks")
    os.makedirs(save_path_img, exist_ok=True)
    os.makedirs(save_path_mask, exist_ok=True)

    for folder_path in folder_paths:
        if is_test:
            file_id = f"patient0{str(test_count).zfill(4)}"
            test_count += 1
        else:
            file_id = f"patient0{str(train_count).zfill(4)}"
            train_count += 1

        # loop through echocardiogram images of this patient
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # skip non-image files
            if file.endswith(".mhd") and "sequence" not in file:
                image_data, image_header = medpy_io.load(file_path)
                file_suffix = get_camus_file_suffix(file)
                filename = f"{file_id}_{file_suffix}{file_ending}"
                filenames.append(filename)
                img_file_save_path = os.path.join(save_path_img, filename)
                mask_file_save_path = os.path.join(save_path_mask, filename)

                if "_gt" in file:
                    # ground truth data consists of mask with value 1, 2, or 3
                    # replace values for better visibility in images
                    image_data[image_data == 1] = 85  # left ventricle
                    image_data[image_data == 2] = 170  # left ventricle wall
                    image_data[image_data == 3] = 255  # left atrium
                    medpy_io.save(image_data, mask_file_save_path)
                else:
                    medpy_io.save(image_data, img_file_save_path)

    return sorted(filenames)


@beartype
def get_camus_file_suffix(string: str):
    """
    Extracts the camus suffix, i.e. from a file "patient0001_2CH_ED.mhd" it would extract "2CH_ED". Also removes the
    "_gt" in the end if it is present.
    """

    dot_index = string.rfind('.')
    underscore_index = string.find('_')
    if dot_index >= 0:
        string = string[:dot_index]
    if underscore_index >= 0:
        string = string[underscore_index + 1:]
    string = string.replace("_gt", "")
    return string


if __name__ == "__main__":
    convert_camus()
