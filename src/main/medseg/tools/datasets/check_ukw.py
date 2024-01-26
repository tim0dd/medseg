import os
from collections import defaultdict
from typing import Optional

import click
from beartype import beartype
from medseg.verification.datasets.check_for_duplicates import find_similar, create_duplicates_text_summary

from medseg.data.converters.helpers import get_image_dims
from medseg.util.files import save_text_to_file


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(), required=False)
@beartype
def check_ukw(in_path: str, out_path: Optional[str]):
    """
       Check UKW dataset
       Original structure is roughly as follows:
           - ukw (main directory)
              file_name_1.jpg
              file_name_1_mask_cb.JPG
              file_name_2.jpg
              file_name_2_mask_cb.jpg
              file_name_3.jpg
              file_name_3_mask_sh.JPG

       Out folder structure should be:
            - images (directory)
              file_name_1.jpg
              file_name_2.jpg
              file_name_3.jpg

            - masks (directory)
              file_name_1_mask_cb.JPG
              file_name_2_mask_cb.jpg
              file_name_3_mask_sh.jpg

       :param in_path:     Path to the original main directory.
       :param out_path:    Path to the output directory.
       """

    file_names = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    file_names_lower = [f.lower() for f in file_names]

    names_lower_to_original = {}
    for i, fname_lower in enumerate(file_names_lower):
        names_lower_to_original[fname_lower] = file_names[i]

    img_mask_dict = defaultdict(list)
    missing_imgs_derived_from_masks = set()

    for fname_lower, fname in names_lower_to_original.items():
        if "_mask" in fname_lower:
            base_name, ext = os.path.splitext(fname)
            img_name_l = (base_name.split("_mask")[0] + ext).lower()
            if img_name_l not in names_lower_to_original.keys():
                missing_imgs_derived_from_masks.add(img_name_l)
            else:
                img_name = names_lower_to_original[img_name_l]
                img_mask_dict[img_name].append(fname)
        else:
            if fname not in img_mask_dict.keys():
                img_mask_dict[fname] = []

    n_masks_per_image = [len(v) for v in img_mask_dict.values()]
    images_with_missing_masks = [k for k, v in img_mask_dict.items() if len(v) == 0]

    n_images_with_masks = len([k for k, v in img_mask_dict.items() if len(v) > 0])

    output_lines = []
    output_lines.append("---------------------------- Image-Masks filename matching ----------------------------")
    output_lines.append(f"Total number of files: {len(file_names)}")
    output_lines.append(
        f"Found {n_images_with_masks} images and {sum(n_masks_per_image)} masks that have matching masks or image.")
    output_lines.append(f"Found {len(images_with_missing_masks)} images without masks.")
    output_lines.append(f"Found {len(missing_imgs_derived_from_masks)} masks without images.")
    output_lines.append(f"There are at most {max(n_masks_per_image)} masks for one image.")
    output_lines.append(f"There are at least {min(n_masks_per_image)} masks for one image.")

    output_lines.append("Expected to find images with the following names (derived from mask names) but couldn't")
    for missing_img in missing_imgs_derived_from_masks:
        output_lines.append(f"    {missing_img}")

    output_lines.append("Expected to find masks for the following image names (derived from image names) but couldn't:")
    for img_with_missing_mask in images_with_missing_masks:
        output_lines.append(f"    {img_with_missing_mask}")

    output_lines.append("---------------------------- Image-Mask dimension mismatches ----------------------------")
    for img_name, mask_names in img_mask_dict.items():
        # check image dimensions against mask dimensions, find mismatches
        if len(mask_names) > 0:
            img_path = os.path.join(in_path, img_name)
            img_h, img_w = get_image_dims(img_path, 'jpg')
            for mask_name in mask_names:
                mask_path = os.path.join(in_path, mask_name)
                mask_h, mask_w = get_image_dims(mask_path, 'jpg')
                if img_h != mask_h or img_w != mask_w:
                    output_lines.append(
                        f"Image {img_name} has dimensions ({img_h}, {img_w}) but mask {mask_name} has dimensions ({mask_h}, {mask_w})")

    output_lines.append("---------------------------- List of unique image dimensions ----------------------------")
    dimensions_dict = defaultdict(list)
    for fname in file_names:
        h, w = get_image_dims(os.path.join(in_path, fname), 'jpg')
        dimensions_dict[(h, w)].append(fname)

    output_lines.append(f"Found {len(dimensions_dict.keys())} different image dimensions")
    output_lines.append(f"Format:  (H, W): number of times")
    for d, fnames in dimensions_dict.items():
        output_lines.append(f"    {d} : {len(fnames)} times")

    output_lines.append(
        "---------------------------- Check for duplicate files or similar images according to image hash "
        "----------------------------")
    # build all file paths but filter out masks
    img_paths = [os.path.join(in_path, fname) for fname in file_names if "mask" not in fname.lower()]
    output_text = "\n".join(output_lines)
    identical_bytes, identical_hashes, high_similarity = find_similar(img_paths, 11)
    duplicate_summary = create_duplicates_text_summary(identical_bytes, identical_hashes, high_similarity)
    output_text = f"{output_text}\n{duplicate_summary}\n"
    print(output_text)
    if out_path is not None:
        save_text_to_file(output_text, os.path.join(out_path))
        print(f"Saved output to {out_path}")


if __name__ == "__main__":
    check_ukw()
