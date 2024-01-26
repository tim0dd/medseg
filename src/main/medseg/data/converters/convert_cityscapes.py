import os
from typing import Tuple, List

import click
from beartype import beartype

from medseg.data.converters.converters import create_dataset_predefined_split
from medseg.data.converters.helpers import check_in_folder_paths, copy_files_from_to


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
@beartype
def convert_cityscapes(in_path=None, out_path=None, copy_files=True):
    """
    Convert the Cityscapes dataset (fine annotations), not the coarse version
    Expects to find the cominbed images and annotations in same folders
    There are many download options from the Cityscapes website, for this script to work, the
    leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip are the correct files.

    Very important: the annotations have to be converted to the officially used train IDs with the official script first
    Instructions can be found on https://github.com/mcordts/cityscapesScripts
    The script is located in the above repository under cityscapesscripts/preparation/createTrainIdLabelImgs.py
    The result should be additional annotations with the suffix 'gtFine_labelTrainIds'

    The images from leftImg8bit_trainvaltest.zip should be copied into the same folder as the annotations

    The expected structure for the annotations is as follows:
        - train (18 subfolders)
            - aachen
                - aachen_000000_000019_gtFine_color.png
                - aachen_000000_000019_gtFine_instanceIds.png
                - aachen_000000_000019_gtFine_labelIds.png
                - aachen_000012_000019_gtFine_labelTrainIds.png     <------ annotation
                - aachen_000012_000019_gtFine_polygons.json
                - aachen_000000_000019_leftImg8bit.png              <------ image
                ...
            ...
        - val (3 subfolders)
            - frankfurt
                - frankfurt_000000_000294_gtFine_color.png
                - frankfurt_000000_000294_gtFine_instanceIds.png
                - frankfurt_000000_000294_gtFine_labelIds.png
                - frankfurt_000000_000294_gtFine_labelTrainIds.png  <------ annotation
                - frankfurt_000000_000294_gtFine_polygons.json
                - frankfurt_000000_000294_leftImg8bit.png           <------ image
                ...
            ...

    Args:
        in_path (str): Path to the original main directory.
        out_path (str): Path where the converted data should be saved.
        copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
        and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    train_path_in = os.path.join(in_path, "train")
    val_path_in = os.path.join(in_path, "val")
    check_in_folder_paths(train_path_in, val_path_in)
    class_dict = {"background": 255,
                  "road": 0,
                  "sidewalk": 1,
                  "building": 2,
                  "wall": 3,
                  "fence": 4,
                  "pole": 5,
                  "traffic light": 6,
                  "traffic sign": 7,
                  "vegetation": 8,
                  "terrain": 9,
                  "sky": 10,
                  "person": 11,
                  "rider": 12,
                  "car": 13,
                  "truck": 14,
                  "bus": 15,
                  "train": 16,
                  "motorcycle": 17,
                  "bicycle": 18,
                  }

    train_paths_in = collect_cityscapes_image_annotation_tuples(train_path_in)
    val_paths_in = collect_cityscapes_image_annotation_tuples(val_path_in)

    test_paths_in = []
    create_dataset_predefined_split(train_paths_in, val_paths_in, test_paths_in, out_path, class_dict,
                                    exclude_files=None, strict_filename_matching=False, copy_files=copy_files)
    # copy labelId and instanceId images of the validation set too, as they are needed to perform the evaluation
    # using the official cityscapes evaluation script
    for city_folder in os.listdir(val_path_in):
        city_folder_path = os.path.join(val_path_in, city_folder)
        for filename in os.listdir(city_folder_path):
            if filename.endswith('_gtFine_labelIds.png') or filename.endswith('_gtFine_instanceIds.png'):
                copy_files_from_to(os.path.join(city_folder_path, filename), os.path.join(out_path, 'masks'))


@beartype
def collect_cityscapes_image_annotation_tuples(path_in: str) -> List[Tuple[str, str]]:
    image_annotation_pairs = []
    for city_folder in os.listdir(path_in):
        city_folder_path = os.path.join(path_in, city_folder)
        for filename in os.listdir(city_folder_path):
            if filename.endswith('_leftImg8bit.png'):
                # Getting annotation filename corresponding to image
                annotation_filename = filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                image_path = os.path.join(path_in, city_folder_path, filename)
                annotation_path = os.path.join(path_in, city_folder_path, annotation_filename)
                if os.path.exists(annotation_path):
                    image_annotation_pairs.append((image_path, annotation_path))
                else:
                    raise FileNotFoundError(f"Annotation file {annotation_filename} not found for image {filename}")
    return image_annotation_pairs


if __name__ == "__main__":
    convert_cityscapes()
