import collections
import os

import click
import cv2
import numpy as np
import pandas as pd
import skimage.draw
from PIL import Image

from medseg.data.converters.helpers import create_dataset_subfolders
from medseg.data.split_type import SplitType


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=True), required=True)
@click.option("--out_path", type=click.Path(exists=False, file_okay=True), required=True)
def convert_echonet_dynamic(in_path=None, out_path=None):
    """
       Convert the original Echonet Dynamic video files to images, generate masks and add an index.csv and classes.csv file.
       Original structure is as follows:
        - Videos (10030 video files in avi format)
           - 0X1A0A263B22CCD966.avi
           - ...
           - 0XFEBEEFF93F6FEB9.avi
        - FileList.csv (contains the file names and meta information)
        - VolumeTracings.csv (contains the tracings to calculate the segmentation masks for the left ventricle)

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.

    Returns:
    None
       """
    with open(os.path.join(in_path, "FileList.csv")) as f:
        data = pd.read_csv(f)
    file_names, ids, splits = filter_missing_files(data, in_path)
    os.makedirs(out_path, exist_ok=True)

    img_path_out, mask_path_out = create_dataset_subfolders(out_path)
    trace = load_trace(in_path)
    rows = []
    for i, file_name in enumerate(file_names):
        video = load_video(os.path.join(in_path, "Videos", file_name))
        if video is not None:
            frame_ed, frame_es = find_ed_and_es_frames(trace, file_name)
            if frame_ed is not None and frame_es is not None:
                split = SplitType(splits[i].lower())
                ed = video[frame_ed]
                ed_mask = generate_ground_truth(trace[file_name][frame_ed])
                es = video[frame_es]
                es_mask = generate_ground_truth(trace[file_name][frame_es])
                ed_row = save_echo_img_mask(ed, ed_mask, img_path_out, mask_path_out, ids[i], False, split)
                es_row = save_echo_img_mask(es, es_mask, img_path_out, mask_path_out, ids[i], True, split)
                rows.append(ed_row)
                rows.append(es_row)

    # create index dataframe and save to csv
    df = pd.DataFrame(rows, columns=["image", "mask", "split", "height", "width", "img_type", "mask_type"])
    df.to_csv(os.path.join(out_path, "index.csv"), index=True)
    class_list = [["background", 0], ["lv", 255]]
    df = pd.DataFrame(class_list, columns=["label", "pixel_value"])
    df.to_csv(os.path.join(out_path, "classes.csv"), index=True)


def save_echo_img_mask(img: np.ndarray, mask: np.ndarray, img_path_out: str, mask_path_out: str, sample_id: str,
                       is_es: bool, split: SplitType) -> list:
    mask_suffix = "ES_mask" if is_es else "ED_mask"
    img_suffix = "ES" if is_es else "ED"
    mask_name = f"{sample_id}_{mask_suffix}.png"
    img_name = f"{sample_id}_{img_suffix}.png"
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    img.save(os.path.join(img_path_out, img_name))
    mask.save(os.path.join(mask_path_out, mask_name))
    if img.height != mask.height or img.width != mask.width:
        raise Warning(f"Image and mask dimensions do not match for files {img_name} and {mask_name}")
    row = [img_name, mask_name, split.value, img.height, img.width, "png", "png"]
    return row


def filter_missing_files(data, in_path):
    fnames = [f"{fn}.avi" if os.path.splitext(fn)[1] == "" else fn for fn in data["FileName"].tolist()]
    missing = set(fnames) - set(os.listdir(os.path.join(in_path, "Videos")))

    if missing:
        print(f"{len(missing)} videos could not be found in {os.path.join(in_path, 'Videos')}:")
        for f in sorted(missing):
            print("\t", f)
        raise FileNotFoundError(os.path.join(in_path, "Videos", sorted(missing)[0]))

    return fnames, data["FileName"].tolist(), data["Split"].str.upper().tolist()


def load_trace(in_path):
    with open(os.path.join(in_path, "VolumeTracings.csv")) as f:
        header = f.readline().strip().split(",")
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

        trace = collections.defaultdict(lambda: collections.defaultdict(list))
        for line in f:
            filename, x1, y1, x2, y2, frame = line.strip().split(',')
            trace[filename][int(frame)].append((float(x1), float(y1), float(x2), float(y2)))

    return trace


def load_video(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        print("#### WARNING_NOT_FOUND: " + filename)
        return None

    capture = cv2.VideoCapture(filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    video = np.zeros((frame_count, frame_height, frame_width), np.uint8)
    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError(f"Failed to load frame #{count} of {filename}")

        video[count] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return video


def find_ed_and_es_frames(trace, filename):
    frames = sorted(trace[filename].keys())
    if len(frames) >= 2:
        frame_es, frame_ed = frames[:2]
    else:
        print(f"Insufficient frames for {filename}")
        frame_es, frame_ed = None, None
    return frame_ed, frame_es


def generate_ground_truth(trace) -> np.ndarray:
    height, width, mask_value = 112, 112, 255

    trace = np.array(trace)
    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))

    r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (height, width))
    mask = np.zeros((height, width), np.uint8)
    mask[r, c] = mask_value
    return mask


if __name__ == "__main__":
    convert_echonet_dynamic()
