import unittest

import numpy as np
import torch
from PIL import Image
from medseg.util.img_ops import calculate_mean_std, get_pil_mode
from torchvision.transforms import ToTensor


class TestImgOps(unittest.TestCase):

    def test_calculate_mean_std(self):
        num_images = 1000
        size = 250
        n_channels = 3
        use_to_tensor_transform = True
        dataset = MockDataset(num_images, size, n_channels, use_to_tensor_transform)
        mean, std = calculate_mean_std(dataset.get_image, num_images, 3)

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertEqual(mean.dtype, torch.float32)
        self.assertEqual(std.dtype, torch.float32)
        # Check if mean and std are within expected bounds
        self.assertTrue(torch.all(mean >= 0) and torch.all(mean <= 1))
        self.assertTrue(torch.all(std >= 0) and torch.all(std <= 1))
        # means should be very close to 0.5 and stds should be very close to 0.2887
        self.assertTrue(torch.all(mean >= 0.497) and torch.all(mean <= 0.503))
        self.assertTrue(torch.all(std >= 0.287) and torch.all(std <= 0.289))


class MockDataset:
    def __init__(self, num_images, size, n_channels, use_to_tensor_transform):
        self.num_images = num_images
        self.size = size
        self.n_channels = n_channels
        self.use_to_tensor_transform = use_to_tensor_transform
        self.shape = (self.size, self.size, self.n_channels)
        self.pil_mode = get_pil_mode(self.n_channels)

    def __len__(self):
        return self.num_images

    def get_image(self, index):
        if index >= self.num_images:
            raise IndexError("Index out of range")
        img = np.random.randint(0, 255, size=self.shape, dtype=np.uint8)
        image = Image.fromarray(img, mode=self.pil_mode)
        if self.use_to_tensor_transform:
            image = ToTensor()(image)
        return image


if __name__ == '__main__':
    unittest.main()
