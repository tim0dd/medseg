import os
import unittest

import numpy as np
import torch
from PIL import Image, ImageChops
from torch import Tensor
from torchvision import transforms as t
from torchvision.transforms import InterpolationMode

from medseg.config.config import load_and_parse_config
from medseg.data.split_type import SplitType
from medseg.data.transforms.transforms import MedsegTransform


class TransformsTest(unittest.TestCase):
    def test_color_jitter(self):
        cfg = self._load_transform_test_config("config_test_color_jitter.yaml")
        color_jitter_train = cfg["transforms"][SplitType.TRAIN.value][0]
        color_jitter_val = cfg["transforms"][SplitType.VAL.value][0]
        color_jitter_test = cfg["transforms"][SplitType.TEST.value][0]
        self.assertTrue("color_jitter" in color_jitter_train)
        self.assertTrue("color_jitter" in color_jitter_val)
        self.assertTrue("color_jitter" in color_jitter_test)
        color_jitter_train_args = {"brightness": (0.1, 0.2), "contrast": (0.3, 0.4), "saturation": (0.5, 0.6),
                                   "hue": (-0.4, 0.4)}
        color_jitter_val_args = {"brightness": 0.1, "contrast": 0.2, "saturation": 0.3, "hue": 0.4}
        color_jitter_test_args = {"brightness": 0.5, "contrast": 0.6, "saturation": 0.7, "hue": 0.5}
        self.assertEqual(color_jitter_train["color_jitter"].args, color_jitter_train_args)
        self.assertEqual(color_jitter_val["color_jitter"].args, color_jitter_val_args)
        self.assertEqual(color_jitter_test["color_jitter"].args, color_jitter_test_args)
        color_jitter_train["color_jitter"].compose()
        color_jitter_val["color_jitter"].compose()
        color_jitter_test["color_jitter"].compose()
        built_transform_train = color_jitter_train["color_jitter"].img_transforms.transforms[0]
        built_transform_val = color_jitter_val["color_jitter"].img_transforms.transforms[0]
        built_transform_test = color_jitter_test["color_jitter"].img_transforms.transforms[0]
        self.assertIsInstance(built_transform_train, t.ColorJitter)
        self.assertIsInstance(built_transform_val, t.ColorJitter)
        self.assertIsInstance(built_transform_test, t.ColorJitter)
        self.assertEqual(built_transform_train.brightness, color_jitter_train_args["brightness"])
        self.assertEqual(built_transform_train.contrast, color_jitter_train_args["contrast"])
        self.assertEqual(built_transform_train.saturation, color_jitter_train_args["saturation"])
        self.assertEqual(built_transform_train.hue, color_jitter_train_args["hue"])
        val_brightness_min = 1 - color_jitter_val_args["brightness"]
        val_brightness_max = 1 + color_jitter_val_args["brightness"]
        self.assertEqual(built_transform_val.brightness, (val_brightness_min, val_brightness_max))
        val_contrast_min = 1 - color_jitter_val_args["contrast"]
        val_contrast_max = 1 + color_jitter_val_args["contrast"]
        self.assertEqual(built_transform_val.contrast, (val_contrast_min, val_contrast_max))
        val_saturation_min = 1 - color_jitter_val_args["saturation"]
        val_saturation_max = 1 + color_jitter_val_args["saturation"]
        self.assertEqual(built_transform_val.saturation, (val_saturation_min, val_saturation_max))
        val_hue_min = -color_jitter_val_args["hue"]
        val_hue_max = color_jitter_val_args["hue"]
        self.assertEqual(built_transform_val.hue, (val_hue_min, val_hue_max))
        test_brightness_min = 1 - color_jitter_test_args["brightness"]
        test_brightness_max = 1 + color_jitter_test_args["brightness"]
        self.assertEqual(built_transform_test.brightness, (test_brightness_min, test_brightness_max))
        test_contrast_min = 1 - color_jitter_test_args["contrast"]
        test_contrast_max = 1 + color_jitter_test_args["contrast"]
        self.assertEqual(built_transform_test.contrast, (test_contrast_min, test_contrast_max))
        test_saturation_min = 1 - color_jitter_test_args["saturation"]
        test_saturation_max = 1 + color_jitter_test_args["saturation"]
        self.assertEqual(built_transform_test.saturation, (test_saturation_min, test_saturation_max))
        test_hue_min = -color_jitter_test_args["hue"]
        test_hue_max = color_jitter_test_args["hue"]
        self.assertEqual(built_transform_test.hue, (test_hue_min, test_hue_max))
        self.assertTrue(color_jitter_train["color_jitter"].mask_transforms is None)
        self.assertTrue(color_jitter_val["color_jitter"].mask_transforms is None)
        self.assertTrue(color_jitter_test["color_jitter"].mask_transforms is None)
        self._check_if_transformed(color_jitter_train["color_jitter"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(color_jitter_train["color_jitter"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(color_jitter_val["color_jitter"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(color_jitter_val["color_jitter"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(color_jitter_test["color_jitter"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(color_jitter_test["color_jitter"], pil_image=False, is_size_altered=False)

    def test_gaussian_blur(self):
        cfg = self._load_transform_test_config("config_test_gaussian_blur.yaml")
        gaussian_train = cfg["transforms"][SplitType.TRAIN.value][0]
        gaussian_val = cfg["transforms"][SplitType.VAL.value][0]
        gaussian_test = cfg["transforms"][SplitType.TEST.value][0]
        self.assertTrue("gaussian_blur" in gaussian_train)
        self.assertTrue("gaussian_blur" in gaussian_val)
        self.assertTrue("gaussian_blur" in gaussian_test)
        gaussian_train_args = {"kernel_size": 11, "sigma": (10, 20)}
        gaussian_val_args = {"kernel_size": 13, "sigma": (30, 40)}
        gaussian_test_args = {"kernel_size": 15, "sigma": (50, 60)}
        self.assertEqual(gaussian_train["gaussian_blur"].args, gaussian_train_args)
        self.assertEqual(gaussian_val["gaussian_blur"].args, gaussian_val_args)
        self.assertEqual(gaussian_test["gaussian_blur"].args, gaussian_test_args)
        gaussian_train["gaussian_blur"].compose()
        gaussian_val["gaussian_blur"].compose()
        gaussian_test["gaussian_blur"].compose()
        built_transform_train = gaussian_train["gaussian_blur"].img_transforms.transforms[0]
        built_transform_val = gaussian_val["gaussian_blur"].img_transforms.transforms[0]
        built_transform_test = gaussian_test["gaussian_blur"].img_transforms.transforms[0]
        self.assertIsInstance(built_transform_train, t.GaussianBlur)
        self.assertIsInstance(built_transform_val, t.GaussianBlur)
        self.assertIsInstance(built_transform_test, t.GaussianBlur)
        self.assertEqual(built_transform_train.kernel_size, (gaussian_train_args["kernel_size"],
                                                             gaussian_train_args["kernel_size"]))
        self.assertEqual(built_transform_val.kernel_size, (gaussian_val_args["kernel_size"],
                                                           gaussian_val_args["kernel_size"]))
        self.assertEqual(built_transform_test.kernel_size,
                         (gaussian_test_args["kernel_size"], gaussian_test_args["kernel_size"]))
        self.assertEqual(built_transform_train.sigma, gaussian_train_args["sigma"])
        self.assertEqual(built_transform_val.sigma, gaussian_val_args["sigma"])
        self.assertEqual(built_transform_test.sigma, gaussian_test_args["sigma"])
        self.assertTrue(gaussian_train["gaussian_blur"].mask_transforms is None)
        self.assertTrue(gaussian_val["gaussian_blur"].mask_transforms is None)
        self.assertTrue(gaussian_test["gaussian_blur"].mask_transforms is None)
        self._check_if_transformed(gaussian_train["gaussian_blur"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(gaussian_train["gaussian_blur"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(gaussian_val["gaussian_blur"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(gaussian_val["gaussian_blur"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(gaussian_test["gaussian_blur"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(gaussian_test["gaussian_blur"], pil_image=False, is_size_altered=False)

    def test_normalize(self):
        cfg = self._load_transform_test_config("config_test_normalize.yaml")
        normalize_train = cfg["transforms"][SplitType.TRAIN.value][0]
        normalize_val = cfg["transforms"][SplitType.VAL.value][0]
        normalize_test = cfg["transforms"][SplitType.TEST.value][0]
        self.assertTrue("normalize" in normalize_train)
        self.assertTrue("normalize" in normalize_val)
        self.assertTrue("normalize" in normalize_test)
        normalize_train_args = {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}
        normalize_val_args = {"mean": [0.4, 0.5, 0.6], "std": [0.7, 0.8, 0.9]}
        normalize_test_args = {"mean": [0.6, 0.5, 0.4], "std": [0.3, 0.2, 0.1]}
        self.assertEqual(normalize_train["normalize"].args, normalize_train_args)
        self.assertEqual(normalize_val["normalize"].args, normalize_val_args)
        self.assertEqual(normalize_test["normalize"].args, normalize_test_args)
        normalize_train["normalize"].compose()
        normalize_val["normalize"].compose()
        normalize_test["normalize"].compose()
        built_transform_train = normalize_train["normalize"].img_transforms.transforms[0]
        built_transform_val = normalize_val["normalize"].img_transforms.transforms[0]
        built_transform_test = normalize_test["normalize"].img_transforms.transforms[0]
        self.assertIsInstance(built_transform_train, t.Normalize)
        self.assertIsInstance(built_transform_val, t.Normalize)
        self.assertIsInstance(built_transform_test, t.Normalize)
        self.assertEqual(built_transform_train.mean, normalize_train_args["mean"])
        self.assertEqual(built_transform_val.mean, normalize_val_args["mean"])
        self.assertEqual(built_transform_test.mean, normalize_test_args["mean"])
        self.assertEqual(built_transform_train.std, normalize_train_args["std"])
        self.assertEqual(built_transform_val.std, normalize_val_args["std"])
        self.assertEqual(built_transform_test.std, normalize_test_args["std"])
        self.assertTrue(normalize_train["normalize"].mask_transforms is None)
        self.assertTrue(normalize_val["normalize"].mask_transforms is None)
        self.assertTrue(normalize_test["normalize"].mask_transforms is None)
        # normalize transform only works on tensors, not on PIL images
        self._check_if_transformed(normalize_train["normalize"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(normalize_val["normalize"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(normalize_test["normalize"], pil_image=False, is_size_altered=False)

    def test_pad_and_resize(self):
        cfg = self._load_transform_test_config("config_test_pad_and_resize.yaml")
        pad_and_resize_train = cfg["transforms"][SplitType.TRAIN.value][0]
        pad_and_resize_val = cfg["transforms"][SplitType.VAL.value][0]
        pad_and_resize_test = cfg["transforms"][SplitType.TEST.value][0]
        self.assertTrue("pad_and_resize" in pad_and_resize_train)
        self.assertTrue("pad_and_resize" in pad_and_resize_val)
        self.assertTrue("pad_and_resize" in pad_and_resize_test)
        pad_and_resize_train_args = {"size": 100, "padding_mode": "edge", "mask_fill": 1, "h": 50, "w": 76}
        pad_and_resize_val_args = {"size": 101, "padding_mode": "constant", "fill": 1, "mask_fill": 2, "h": 77,
                                   "w": 51}
        pad_and_resize_test_args = {"size": 102, "padding_mode": "reflect", "mask_fill": 3, "h": 52, "w": 60}
        self.assertEqual(pad_and_resize_train["pad_and_resize"].args, pad_and_resize_train_args)
        self.assertEqual(pad_and_resize_val["pad_and_resize"].args, pad_and_resize_val_args)
        self.assertEqual(pad_and_resize_test["pad_and_resize"].args, pad_and_resize_test_args)
        pad_and_resize_train["pad_and_resize"].compose()
        pad_and_resize_val["pad_and_resize"].compose()
        pad_and_resize_test["pad_and_resize"].compose()

        # Check types
        built_img_pad_transform_train = pad_and_resize_train["pad_and_resize"].img_transforms.transforms[0]
        built_img_pad_transform_val = pad_and_resize_val["pad_and_resize"].img_transforms.transforms[0]
        built_img_pad_transform_test = pad_and_resize_test["pad_and_resize"].img_transforms.transforms[0]
        self.assertIsInstance(built_img_pad_transform_train, t.Pad)
        self.assertIsInstance(built_img_pad_transform_val, t.Pad)
        self.assertIsInstance(built_img_pad_transform_test, t.Pad)
        built_img_resize_transform_train = pad_and_resize_train["pad_and_resize"].img_transforms.transforms[1]
        built_img_resize_transform_val = pad_and_resize_val["pad_and_resize"].img_transforms.transforms[1]
        built_img_resize_transform_test = pad_and_resize_test["pad_and_resize"].img_transforms.transforms[1]
        self.assertIsInstance(built_img_resize_transform_train, t.Resize)
        self.assertIsInstance(built_img_resize_transform_val, t.Resize)
        self.assertIsInstance(built_img_resize_transform_test, t.Resize)
        built_mask_pad_transform_train = pad_and_resize_train["pad_and_resize"].mask_transforms.transforms[0]
        built_mask_pad_transform_val = pad_and_resize_val["pad_and_resize"].mask_transforms.transforms[0]
        built_mask_pad_transform_test = pad_and_resize_test["pad_and_resize"].mask_transforms.transforms[0]
        self.assertIsInstance(built_mask_pad_transform_train, t.Pad)
        self.assertIsInstance(built_mask_pad_transform_val, t.Pad)
        self.assertIsInstance(built_mask_pad_transform_test, t.Pad)
        built_mask_resize_transform_train = pad_and_resize_train["pad_and_resize"].mask_transforms.transforms[1]
        built_mask_resize_transform_val = pad_and_resize_val["pad_and_resize"].mask_transforms.transforms[1]
        built_mask_resize_transform_test = pad_and_resize_test["pad_and_resize"].mask_transforms.transforms[1]
        self.assertIsInstance(built_mask_resize_transform_train, t.Resize)
        self.assertIsInstance(built_mask_resize_transform_val, t.Resize)
        self.assertIsInstance(built_mask_resize_transform_test, t.Resize)

        # Check img transforms args
        target_padding_train = (0, 13, 0, 13)  # left, top, right, bottom
        target_padding_val = (13, 0, 13, 0)
        target_padding_test = (0, 4, 0, 4)
        self.assertEqual(built_img_pad_transform_train.padding, target_padding_train)
        self.assertEqual(built_img_pad_transform_val.padding, target_padding_val)
        self.assertEqual(built_img_pad_transform_test.padding, target_padding_test)

        self.assertEqual(built_img_pad_transform_train.padding_mode, "edge")
        self.assertEqual(built_img_pad_transform_val.padding_mode, "constant")
        self.assertEqual(built_img_pad_transform_test.padding_mode, "reflect")

        self.assertEqual(built_img_pad_transform_train.fill, 0)
        self.assertEqual(built_img_pad_transform_val.fill, 1)
        self.assertEqual(built_img_pad_transform_test.fill, 0)

        self.assertEqual(built_img_resize_transform_train.size, (100, 100))
        self.assertEqual(built_img_resize_transform_val.size, (101, 101))
        self.assertEqual(built_img_resize_transform_test.size, (102, 102))

        self.assertEqual(built_img_resize_transform_train.interpolation, InterpolationMode.BILINEAR)
        self.assertEqual(built_img_resize_transform_val.interpolation, InterpolationMode.BILINEAR)
        self.assertEqual(built_img_resize_transform_test.interpolation, InterpolationMode.BILINEAR)

        # Check mask transforms args
        self.assertEqual(built_mask_pad_transform_train.padding, target_padding_train)
        self.assertEqual(built_mask_pad_transform_val.padding, target_padding_val)
        self.assertEqual(built_mask_pad_transform_test.padding, target_padding_test)

        # padding mode for masks should always be constant!
        self.assertEqual(built_mask_pad_transform_train.padding_mode, "constant")
        self.assertEqual(built_mask_pad_transform_val.padding_mode, "constant")
        self.assertEqual(built_mask_pad_transform_test.padding_mode, "constant")

        self.assertEqual(built_mask_pad_transform_train.fill, 1)
        self.assertEqual(built_mask_pad_transform_val.fill, 2)
        self.assertEqual(built_mask_pad_transform_test.fill, 3)

        self.assertEqual(built_mask_resize_transform_train.size, (100, 100))
        self.assertEqual(built_mask_resize_transform_val.size, (101, 101))
        self.assertEqual(built_mask_resize_transform_test.size, (102, 102))

        self.assertEqual(built_mask_resize_transform_train.interpolation, InterpolationMode.NEAREST)
        self.assertEqual(built_mask_resize_transform_val.interpolation, InterpolationMode.NEAREST)
        self.assertEqual(built_mask_resize_transform_test.interpolation, InterpolationMode.NEAREST)

        self._check_if_transformed(pad_and_resize_train["pad_and_resize"], pil_image=True, is_size_altered=True)
        self._check_if_transformed(pad_and_resize_val["pad_and_resize"], pil_image=True, is_size_altered=True)
        self._check_if_transformed(pad_and_resize_test["pad_and_resize"], pil_image=True, is_size_altered=True)
        self._check_if_transformed(pad_and_resize_train["pad_and_resize"], pil_image=False, is_size_altered=True)
        self._check_if_transformed(pad_and_resize_val["pad_and_resize"], pil_image=False, is_size_altered=True)
        self._check_if_transformed(pad_and_resize_test["pad_and_resize"], pil_image=False, is_size_altered=True)

    def test_random_affine(self):
        cfg = self._load_transform_test_config("config_test_random_affine.yaml")
        random_affine_train = cfg["transforms"][SplitType.TRAIN.value][0]
        random_affine_val = cfg["transforms"][SplitType.VAL.value][0]
        random_affine_test = cfg["transforms"][SplitType.TEST.value][0]
        self.assertTrue("random_affine" in random_affine_train)
        self.assertTrue("random_affine" in random_affine_val)
        self.assertTrue("random_affine" in random_affine_test)
        random_affine_train_args = {"degrees": 11, "translate": (0.2, 0.3), "shear": 40,
                                    "interpolation": InterpolationMode.BILINEAR,
                                    "fill": 0}
        random_affine_val_args = {"degrees": 12, "translate": (0.3, 0.4), "shear": 41,
                                  "interpolation": InterpolationMode.BILINEAR,
                                  "fill": 0}
        random_affine_test_args = {"degrees": 13, "translate": (0.4, 0.5), "shear": 42,
                                   "interpolation": InterpolationMode.BILINEAR,
                                   "fill": 0}
        self.assertEqual(random_affine_train["random_affine"].args, random_affine_train_args)
        self.assertEqual(random_affine_val["random_affine"].args, random_affine_val_args)
        self.assertEqual(random_affine_test["random_affine"].args, random_affine_test_args)
        random_affine_train["random_affine"].compose()
        random_affine_val["random_affine"].compose()
        random_affine_test["random_affine"].compose()
        built_transform_train_img = random_affine_train["random_affine"].img_transforms.transforms[0]
        built_transform_val_img = random_affine_val["random_affine"].img_transforms.transforms[0]
        built_transform_test_img = random_affine_test["random_affine"].img_transforms.transforms[0]
        built_transform_train_mask = random_affine_train["random_affine"].mask_transforms.transforms[0]
        built_transform_val_mask = random_affine_val["random_affine"].mask_transforms.transforms[0]
        built_transform_test_mask = random_affine_test["random_affine"].mask_transforms.transforms[0]

        # Check type
        self.assertIsInstance(built_transform_train_img, t.RandomAffine)
        self.assertIsInstance(built_transform_val_img, t.RandomAffine)
        self.assertIsInstance(built_transform_test_img, t.RandomAffine)
        self.assertIsInstance(built_transform_train_mask, t.RandomAffine)
        self.assertIsInstance(built_transform_val_mask, t.RandomAffine)
        self.assertIsInstance(built_transform_test_mask, t.RandomAffine)

        # Check if the degrees are set correctly
        degrees_train = [-random_affine_train_args["degrees"], random_affine_train_args["degrees"]]
        degrees_val = [-random_affine_val_args["degrees"], random_affine_val_args["degrees"]]
        degrees_test = [-random_affine_test_args["degrees"], random_affine_test_args["degrees"]]
        self.assertEqual(built_transform_train_img.degrees, degrees_train)
        self.assertEqual(built_transform_val_img.degrees, degrees_val)
        self.assertEqual(built_transform_test_img.degrees, degrees_test)
        self.assertEqual(built_transform_train_mask.degrees, degrees_train)
        self.assertEqual(built_transform_val_mask.degrees, degrees_val)
        self.assertEqual(built_transform_test_mask.degrees, degrees_test)

        # Check if translate is set correctly
        self.assertEqual(built_transform_train_img.translate, random_affine_train_args["translate"])
        self.assertEqual(built_transform_val_img.translate, random_affine_val_args["translate"])
        self.assertEqual(built_transform_test_img.translate, random_affine_test_args["translate"])
        self.assertEqual(built_transform_train_mask.translate, random_affine_train_args["translate"])
        self.assertEqual(built_transform_val_mask.translate, random_affine_val_args["translate"])
        self.assertEqual(built_transform_test_mask.translate, random_affine_test_args["translate"])

        # Check if shear is set correctly
        shear_train = [-random_affine_train_args["shear"], random_affine_train_args["shear"]]
        shear_val = [-random_affine_val_args["shear"], random_affine_val_args["shear"]]
        shear_test = [-random_affine_test_args["shear"], random_affine_test_args["shear"]]
        self.assertEqual(built_transform_train_img.shear, shear_train)
        self.assertEqual(built_transform_val_img.shear, shear_val)
        self.assertEqual(built_transform_test_img.shear, shear_test)
        self.assertEqual(built_transform_train_mask.shear, shear_train)
        self.assertEqual(built_transform_val_mask.shear, shear_val)
        self.assertEqual(built_transform_test_mask.shear, shear_test)

        # Check if interpolation is set correctly
        img_inter = random_affine_train_args["interpolation"]
        mask_inter = InterpolationMode.NEAREST
        self.assertEqual(built_transform_train_img.interpolation, img_inter)
        self.assertEqual(built_transform_val_img.interpolation, img_inter)
        self.assertEqual(built_transform_test_img.interpolation, img_inter)
        self.assertEqual(built_transform_train_mask.interpolation, mask_inter)
        self.assertEqual(built_transform_val_mask.interpolation, mask_inter)
        self.assertEqual(built_transform_test_mask.interpolation, mask_inter)

        # Check if fill is set correctly
        self.assertEqual(built_transform_train_img.fill, random_affine_train_args["fill"])
        self.assertEqual(built_transform_val_img.fill, random_affine_val_args["fill"])
        self.assertEqual(built_transform_test_img.fill, random_affine_test_args["fill"])
        self.assertEqual(built_transform_train_mask.fill, random_affine_train_args["fill"])
        self.assertEqual(built_transform_val_mask.fill, random_affine_val_args["fill"])
        self.assertEqual(built_transform_test_mask.fill, random_affine_test_args["fill"])

        self._check_if_transformed(random_affine_train["random_affine"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(random_affine_train["random_affine"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(random_affine_val["random_affine"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(random_affine_val["random_affine"], pil_image=False, is_size_altered=False)
        self._check_if_transformed(random_affine_test["random_affine"], pil_image=True, is_size_altered=False)
        self._check_if_transformed(random_affine_test["random_affine"], pil_image=False, is_size_altered=False)

    def test_random_horizontal_flip(self):
        cfg = self._load_transform_test_config("config_test_random_horizontal_flip.yaml")
        random_horizontal_flip_train = cfg["transforms"][SplitType.TRAIN.value][0]
        random_horizontal_flip_val = cfg["transforms"][SplitType.VAL.value][0]
        random_horizontal_flip_test = cfg["transforms"][SplitType.TEST.value][0]
        random_horizontal_flip_train_args = {"p": 0.1}
        random_horizontal_flip_val_args = {"p": 0.2}
        random_horizontal_flip_test_args = {"p": 0.3}
        random_horizontal_flip_train["random_horizontal_flip"].args = random_horizontal_flip_train_args
        random_horizontal_flip_val["random_horizontal_flip"].args = random_horizontal_flip_val_args
        random_horizontal_flip_test["random_horizontal_flip"].args = random_horizontal_flip_test_args
        random_horizontal_flip_train["random_horizontal_flip"].compose()
        random_horizontal_flip_val["random_horizontal_flip"].compose()
        random_horizontal_flip_test["random_horizontal_flip"].compose()
        built_transform_train = random_horizontal_flip_train["random_horizontal_flip"].img_transforms.transforms[0]
        built_transform_val = random_horizontal_flip_val["random_horizontal_flip"].img_transforms.transforms[0]
        built_transform_test = random_horizontal_flip_test["random_horizontal_flip"].img_transforms.transforms[0]
        self.assertEqual(built_transform_train.p, random_horizontal_flip_train_args["p"])
        self.assertEqual(built_transform_val.p, random_horizontal_flip_val_args["p"])
        self.assertEqual(built_transform_test.p, random_horizontal_flip_test_args["p"])
        new_args = {"p": 1.0}  # override p to test if transform is applied
        self._check_if_transformed(random_horizontal_flip_train["random_horizontal_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_horizontal_flip_train["random_horizontal_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_horizontal_flip_val["random_horizontal_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_horizontal_flip_val["random_horizontal_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_horizontal_flip_test["random_horizontal_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_horizontal_flip_test["random_horizontal_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)

    def test_random_photometric_distort(self):
        cfg = self._load_transform_test_config("config_test_random_photometric_distort.yaml")
        random_photometric_distort_train = cfg["transforms"][SplitType.TRAIN.value][0]
        random_photometric_distort_val = cfg["transforms"][SplitType.VAL.value][0]
        random_photometric_distort_test = cfg["transforms"][SplitType.TEST.value][0]
        random_photometric_distort_train_args = {"brightness": (0.9, 1.1),
                                                 "contrast": (0.9, 1.1),
                                                 "saturation": (0.9, 1.1),
                                                 "hue": (-0.09, 0.09),
                                                 "p": 0.5}
        random_photometric_distort_val_args = {"brightness": (0.8, 1.2),
                                               "contrast": (0.8, 1.2),
                                               "saturation": (0.8, 1.2),
                                               "hue": (-0.08, 0.08),
                                               "p": 0.6}
        random_photometric_distort_test_args = {"brightness": (0.7, 1.3),
                                                "contrast": (0.7, 1.3),
                                                "saturation": (0.7, 1.3),
                                                "hue": (-0.07, 0.07),
                                                "p": 0.7}
        self.assertEqual(random_photometric_distort_train["random_photometric_distort"].args,
                         random_photometric_distort_train_args)
        self.assertEqual(random_photometric_distort_val["random_photometric_distort"].args,
                         random_photometric_distort_val_args)
        self.assertEqual(random_photometric_distort_test["random_photometric_distort"].args,
                         random_photometric_distort_test_args)
        random_photometric_distort_train["random_photometric_distort"].compose()
        random_photometric_distort_val["random_photometric_distort"].compose()
        random_photometric_distort_test["random_photometric_distort"].compose()
        built_transform_train = \
            random_photometric_distort_train["random_photometric_distort"].img_transforms.transforms[0]
        built_transform_val = random_photometric_distort_val["random_photometric_distort"].img_transforms.transforms[0]
        built_transform_test = random_photometric_distort_test["random_photometric_distort"].img_transforms.transforms[
            0]
        self.assertEqual(built_transform_train.brightness, random_photometric_distort_train_args["brightness"])
        self.assertEqual(built_transform_train.contrast, random_photometric_distort_train_args["contrast"])
        self.assertEqual(built_transform_train.saturation, random_photometric_distort_train_args["saturation"])
        self.assertEqual(built_transform_train.hue, random_photometric_distort_train_args["hue"])
        self.assertEqual(built_transform_val.brightness, random_photometric_distort_val_args["brightness"])
        self.assertEqual(built_transform_val.contrast, random_photometric_distort_val_args["contrast"])
        self.assertEqual(built_transform_val.saturation, random_photometric_distort_val_args["saturation"])
        self.assertEqual(built_transform_val.hue, random_photometric_distort_val_args["hue"])
        self.assertEqual(built_transform_test.brightness, random_photometric_distort_test_args["brightness"])
        self.assertEqual(built_transform_test.contrast, random_photometric_distort_test_args["contrast"])
        self.assertEqual(built_transform_test.saturation, random_photometric_distort_test_args["saturation"])
        self.assertEqual(built_transform_test.hue, random_photometric_distort_test_args["hue"])
        self.assertEqual(built_transform_train.p, random_photometric_distort_train_args["p"])
        self.assertEqual(built_transform_val.p, random_photometric_distort_val_args["p"])
        self.assertEqual(built_transform_test.p, random_photometric_distort_test_args["p"])
        new_args = {"p": 1.0}  # override p to test if transform is applied
        self._check_if_transformed(random_photometric_distort_train["random_photometric_distort"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_photometric_distort_train["random_photometric_distort"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_photometric_distort_val["random_photometric_distort"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_photometric_distort_val["random_photometric_distort"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_photometric_distort_test["random_photometric_distort"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_photometric_distort_test["random_photometric_distort"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)

    def test_random_vertical_flip(self):
        cfg = self._load_transform_test_config("config_test_random_vertical_flip.yaml")
        random_vertical_flip_train = cfg["transforms"][SplitType.TRAIN.value][0]
        random_vertical_flip_val = cfg["transforms"][SplitType.VAL.value][0]
        random_vertical_flip_test = cfg["transforms"][SplitType.TEST.value][0]
        random_vertical_flip_train_args = {"p": 0.1}
        random_vertical_flip_val_args = {"p": 0.2}
        random_vertical_flip_test_args = {"p": 0.3}
        random_vertical_flip_train["random_vertical_flip"].args = random_vertical_flip_train_args
        random_vertical_flip_val["random_vertical_flip"].args = random_vertical_flip_val_args
        random_vertical_flip_test["random_vertical_flip"].args = random_vertical_flip_test_args
        random_vertical_flip_train["random_vertical_flip"].compose()
        random_vertical_flip_val["random_vertical_flip"].compose()
        random_vertical_flip_test["random_vertical_flip"].compose()
        built_transform_train = random_vertical_flip_train["random_vertical_flip"].img_transforms.transforms[0]
        built_transform_val = random_vertical_flip_val["random_vertical_flip"].img_transforms.transforms[0]
        built_transform_test = random_vertical_flip_test["random_vertical_flip"].img_transforms.transforms[0]
        self.assertEqual(built_transform_train.p, random_vertical_flip_train_args["p"])
        self.assertEqual(built_transform_val.p, random_vertical_flip_val_args["p"])
        self.assertEqual(built_transform_test.p, random_vertical_flip_test_args["p"])
        new_args = {"p": 1.0}  # override p to test if transform is applied
        self._check_if_transformed(random_vertical_flip_train["random_vertical_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_vertical_flip_train["random_vertical_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_vertical_flip_val["random_vertical_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_vertical_flip_val["random_vertical_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_vertical_flip_test["random_vertical_flip"], pil_image=True,
                                   is_size_altered=False, new_args=new_args)
        self._check_if_transformed(random_vertical_flip_test["random_vertical_flip"], pil_image=False,
                                   is_size_altered=False, new_args=new_args)

    def test_to_tensor(self):
        cfg = self._load_transform_test_config("config_test_to_tensor.yaml")
        train_transforms = cfg["transforms"][SplitType.TRAIN.value]
        val_transforms = cfg["transforms"][SplitType.VAL.value]
        test_transforms = cfg["transforms"][SplitType.TEST.value]
        to_tensor_train = train_transforms[0]
        to_tensor_val = val_transforms[0]
        self.assertTrue(len(test_transforms) == 0)
        to_tensor_train["to_tensor"].compose()
        to_tensor_val["to_tensor"].compose()
        built_transform_train = to_tensor_train["to_tensor"].img_transforms.transforms[0]
        built_transform_val = to_tensor_val["to_tensor"].img_transforms.transforms[0]
        self.assertIsInstance(built_transform_train, t.ToTensor)
        self.assertIsInstance(built_transform_val, t.ToTensor)
        img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
        mask[0, 0] = 255  # make sure that at least one pixel is 255 to test for it later
        img, mask = Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="L")
        img, mask = to_tensor_train["to_tensor"].apply(img, mask)
        self.assertIsInstance(img, Tensor)
        self.assertIsInstance(mask, Tensor)
        self.assertTrue(img.min() >= 0)
        self.assertTrue(img.max() <= 1)
        self.assertEqual(img.shape, (3, 100, 100))
        self.assertTrue(mask.min() >= 0)
        self.assertTrue(mask.max() == 255)  # masks should not be normalized
        self.assertEqual(mask[0, 0, 0], 255)
        self.assertEqual(mask.shape, (1, 100, 100))

    def test_to_pil_image(self):
        cfg = self._load_transform_test_config("config_test_to_pil_image.yaml")
        train_transforms = cfg["transforms"][SplitType.TRAIN.value]
        val_transforms = cfg["transforms"][SplitType.VAL.value]
        test_transforms = cfg["transforms"][SplitType.TEST.value]
        to_pil_image_train = train_transforms[0]
        to_pil_image_val = val_transforms[0]
        self.assertTrue(len(test_transforms) == 0)
        to_pil_image_train["to_pil_image"].compose()
        to_pil_image_val["to_pil_image"].compose()
        built_transform_train = to_pil_image_train["to_pil_image"].img_transforms.transforms[0]
        built_transform_val = to_pil_image_val["to_pil_image"].img_transforms.transforms[0]
        self.assertIsInstance(built_transform_train, t.ToPILImage)
        self.assertIsInstance(built_transform_val, t.ToPILImage)
        img = torch.rand(3, 100, 100, dtype=torch.float32)
        mask = (torch.rand(1, 100, 100, dtype=torch.float32) * 255).type(torch.uint8)
        mask[0, 0, 0] = 255
        img[0, 0, 0] = 1
        img[1, 0, 0] = 1
        img[2, 0, 0] = 1
        img, mask = to_pil_image_train["to_pil_image"].apply(img, mask)
        self.assertIsInstance(img, Image.Image)
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(img.size, (100, 100))
        self.assertEqual(mask.size, (100, 100))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getpixel((0, 0)), 255)
        self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))

    def _check_if_transformed(self, transform: MedsegTransform, pil_image=False, is_size_altered=False, new_args=None):
        is_image_only_transform = transform.apply_to_mask is False
        h = transform.args["h"] if "h" in transform.args else 100
        w = transform.args["w"] if "w" in transform.args else 100
        if pil_image:
            img = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
            mask = np.random.randint(0, 255, size=(h, w), dtype=np.uint8)
            img, mask = Image.fromarray(img, mode="RGB"), Image.fromarray(mask, mode="L")
            img_transformed, mask_transformed = transform.apply(img, mask, new_args)
            self.assertTrue(isinstance(img_transformed, Image.Image))
            self.assertTrue(isinstance(mask_transformed, Image.Image))
            img_diff = ImageChops.difference(img, img_transformed)
            self.assertTrue(img_diff.getbbox() is not None)
            mask_diff = ImageChops.difference(mask, mask_transformed)
            self.assertTrue(mask_diff.getbbox() is None if is_image_only_transform else mask_diff.getbbox() is not None)
            same_img_size = img_transformed.size == img.size
            same_mask_size = mask_transformed.size == mask.size
            self.assertTrue(not same_img_size if is_size_altered else same_img_size)
            self.assertTrue(not same_mask_size if is_size_altered else same_mask_size)

        else:
            img = np.random.randint(0, 255, size=(3, h, w), dtype=np.uint8)
            mask = np.random.randint(0, 255, size=(1, h, w), dtype=np.uint8)
            img, mask = Tensor(img), Tensor(mask)
            img_transformed, mask_transformed = transform.apply(img, mask, new_args)
            self.assertTrue(isinstance(img_transformed, Tensor))
            self.assertTrue(isinstance(mask_transformed, Tensor))
            self.assertTrue(not torch.equal(img_transformed, img))
            mask_equal = torch.equal(mask_transformed, mask)
            self.assertTrue(mask_equal if is_image_only_transform else not mask_equal)
            same_img_size = img_transformed.size() == img.size()
            same_mask_size = mask_transformed.size() == mask.size()
            self.assertTrue(not same_img_size if is_size_altered else same_img_size)
            self.assertTrue(not same_mask_size if is_size_altered else same_mask_size)

    def _load_transform_test_config(self, cfg_name: str) -> dict:
        current_dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(os.path.join(current_dir_path, "transform_test_configs"), cfg_name)
        return load_and_parse_config(file_path)
