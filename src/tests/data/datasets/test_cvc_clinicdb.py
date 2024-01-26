import unittest

import torch
from medseg.data.dataset_manager import DatasetManager
from medseg.data.datasets import CvcClinicDb
from medseg.data.split_type import SplitType


class CvcClinicDbTest(unittest.TestCase):
    def test_cvc_clinicdb(self):
        cfg = {"dataset": {"type": "CvcClinicDb"}, "settings" : {"transforms_backend": "torchvision"}}
        dataset_all = DatasetManager.build_dataset(cfg, SplitType.ALL, include_transforms_manager=True)
        dataset_train = DatasetManager.build_dataset(cfg, SplitType.TRAIN, include_transforms_manager=True)
        dataset_val = DatasetManager.build_dataset(cfg, SplitType.VAL, include_transforms_manager=True)
        dataset_test = DatasetManager.build_dataset(cfg, SplitType.TEST, include_transforms_manager=True)
        self.assertIsInstance(dataset_all, CvcClinicDb)
        self.assertIsInstance(dataset_train, CvcClinicDb)
        self.assertIsInstance(dataset_val, CvcClinicDb)
        self.assertIsInstance(dataset_test, CvcClinicDb)

        self.assertEqual(len(dataset_all), len(dataset_train) + len(dataset_val) + len(dataset_test))
        self.assertEqual(len(dataset_all), 612)
        self.assertEqual(len(dataset_train), 489)
        self.assertEqual(len(dataset_val), 61)
        self.assertEqual(len(dataset_test), 62)

        # test class definitions
        class_defs = dataset_all.get_class_defs()
        self.assertEqual(len(class_defs), 2)
        background_class = next((x for x in class_defs if x["label"].lower() == "background"), None)
        self.assertIsNotNone(background_class)
        self.assertEqual(background_class["pixel_value"], 0)
        self.assertEqual(dataset_all.background_class["label"], "background")
        self.assertEqual(dataset_all.background_class["pixel_value"], 0)
        polyp_class = next((x for x in class_defs if x["label"].lower() == "polyp"), None)
        self.assertIsNotNone(polyp_class)
        self.assertEqual(polyp_class["pixel_value"], 255)
        self.assertEqual(dataset_all.background_class["pixel_value"], 0)

        # test uniqueness of stored image and mask paths
        self.assertEqual(len(dataset_all.image_paths), len(set(dataset_all.image_paths)))
        self.assertEqual(len(dataset_all.mask_paths), len(set(dataset_all.mask_paths)))

        # test that different images are used for the different dataset splits
        self.assertEqual(len(set(dataset_train.image_paths).intersection(set(dataset_val.image_paths))), 0)
        self.assertEqual(len(set(dataset_train.image_paths).intersection(set(dataset_test.image_paths))), 0)
        self.assertEqual(len(set(dataset_val.image_paths).intersection(set(dataset_test.image_paths))), 0)
        # test that different masks are used for the different dataset splits
        self.assertEqual(len(set(dataset_train.mask_paths).intersection(set(dataset_val.mask_paths))), 0)
        self.assertEqual(len(set(dataset_train.mask_paths).intersection(set(dataset_test.mask_paths))), 0)
        self.assertEqual(len(set(dataset_val.mask_paths).intersection(set(dataset_test.mask_paths))), 0)

        # test that images and masks are loaded correctly
        img, mask, sample_id = dataset_all.__getitem__(0)
        self.assertEqual(img.shape, (3, 288, 384))
        self.assertEqual(mask.shape, (1, 288, 384))
        self.assertEqual(mask.max(), 1)
        self.assertEqual(mask.min(), 0)
        self.assertEqual(img.max(), 1)
        self.assertEqual(img.min(), 0)
        self.assertEqual(sample_id, 0)
