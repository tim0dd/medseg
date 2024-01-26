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
def convert_kvasir_cvc_train(in_path=None, out_path=None, copy_files=True):
    """
    Converts the training set from the predefined PraNet/HarDNet-MSEG/HarDNet-DFUS dataset split to enable training
    on the exact same data split as in the original publications. They built their training set from Kvasir-SEG and
    CVC-ClinicDB, however not all files are included as the rest is used for testing. Links for the original data can
    be found in the README.md of https://github.com/DengPingFan/PraNet

    Note that for some reason, while the original data from Kvasir-SEG is in jpg form, this is png.

    Expected folder structure of the original data:
    - image (1450 files)
        1.png, ..., ck2395w2mb4vu07480otsu6tw.png
    - masks (1450 files)
        1.png, ..., ck2395w2mb4vu07480otsu6tw.png

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    mask_path_in = os.path.join(in_path, "masks")
    img_path_in = os.path.join(in_path, "image")
    check_in_folder_paths(mask_path_in, img_path_in)
    train_paths_in = [(img_path_in, mask_path_in)]
    val_paths_in = []
    test_paths_in = []
    class_dict = {"background": 0, "polyp": 255}
    create_dataset_predefined_split(train_paths_in, val_paths_in, test_paths_in, out_path, class_dict,
                                    exclude_files=None, strict_filename_matching=True, copy_files=copy_files)


if __name__ == "__main__":
    convert_kvasir_cvc_train()
