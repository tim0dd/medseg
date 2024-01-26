import os

import click

from medseg.data.converters.converters import create_dataset_predefined_split
from medseg.data.converters.helpers import check_in_folder_paths

KVASIR_CVC_DEFAULT_OUT_PATH = "{project_root}/data/dataset/kvasir_cvc"


# TODO use default path if no arg is given

@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_kvasir_test(in_path=None, out_path=None, copy_files=True):
    """
    Convert the Kvasir-Seg test subset from the predefined PraNet / HarDNet-MSEG / HarDNet-DFUS dataset split to be
    able to evaluate on the exact same data split as in the original publications. Links for the original data can be
    found in the README.md of https://github.com/DengPingFan/PraNet

    Note that for some reason, while the original data from Kvasir-SEG is in jpg form, this is png.

    - images (100 files)
        cju0u82z3cuma0835wlxrnrjv.png, ...
    - masks (100 files)
        cju0u82z3cuma0835wlxrnrjv.png, ...

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
    train_paths_in = [(img_path_in, mask_path_in)]
    val_paths_in = []
    test_paths_in = []
    class_dict = {"background": 0, "polyp": 255}
    create_dataset_predefined_split(train_paths_in, val_paths_in, test_paths_in, out_path, class_dict,
                                    exclude_files=None, strict_filename_matching=True, copy_files=copy_files)


if __name__ == "__main__":
    convert_kvasir_test()
