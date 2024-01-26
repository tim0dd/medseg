import os

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths

CVC_DEFAULT_OUT_PATH = "{project_root}/data/dataset/cvc_clinicdb"


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=False)
def convert_cvc_clinicdb(in_path=None, out_path=None, copy_files=True):
    """
    Convert the original CVC-ClinicDB folder structure and add an index.csv and classes.csv file.
    Original structure is as follows:
    - CVC-ClinicDB
        - Ground Truth (612 files)
            1.tif, 2.tif, ..., 612.tif
        - Original (612 files)
            1.tif, 2.tif, ..., 612.tif

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    mask_path_in = os.path.join(in_path, "Ground Truth")
    img_path_in = os.path.join(in_path, "Original")
    check_in_folder_paths(img_path_in, mask_path_in)
    class_dict = {"background": 0, "polyp": 255}
    folders_in = [(img_path_in, mask_path_in)]
    ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    create_dataset(folders_in, out_path, class_dict, ratios, seed=42, copy_files=copy_files)


if __name__ == "__main__":
    convert_cvc_clinicdb()
