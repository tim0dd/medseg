import os

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths

KVASIR_SEG_DEFAULT_OUT_PATH = "{project_root}/data/dataset/kvasir_seg"


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_kvasir_seg(in_path=None, out_path=None, copy_files=True):
    """
    Convert the original Kvasir-SEG folder structure and add an index.csv and classes.csv file.
    Original structure is as follows:
    - Kvasir-SEG
        - images (1000 files)
            cju0qkwl35piu0993l0dewei2.jpg, ...
        - masks (1000 files)
            cju0qkwl35piu0993l0dewei2.jpg, ...

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    mask_path_in = os.path.join(in_path, "masks")
    img_path_in = os.path.join(in_path, "images")
    check_in_folder_paths(mask_path_in, img_path_in)
    folders_in = [(img_path_in, mask_path_in)]
    class_dict = {"background": 0, "polyp": 255}
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    create_dataset(folders_in, out_path, class_dict, ratios, seed=42, copy_files=copy_files)


if __name__ == "__main__":
    convert_kvasir_seg()
