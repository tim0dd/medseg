import unittest

from medseg.data.class_mapping import ClassMapping


class TestClassMapping(unittest.TestCase):

    def setUp(self):
        self.class_definitions = [
            {"label": "background", "pixel_value": 0},
            {"label": "class1", "pixel_value": 1},
            {"label": "class2", "pixel_value": 2},
        ]
        self.class_mapping = ClassMapping(self.class_definitions)

    def test_initialization(self):
        """Test if the class is initialized correctly with valid class definitions."""
        self.assertEqual(self.class_mapping.background_class, self.class_definitions[0])
        self.assertEqual(self.class_mapping.num_classes, len(self.class_definitions))

    def test_indices_to_pixels_mapping(self):
        """Test if indices to pixels mapping is correctly set up."""
        expected_mapping = {0: 0, 1: 1, 2: 2}
        self.assertEqual(self.class_mapping.indices_to_pixels_mapping, expected_mapping)

    def test_pixels_to_indices_mapping(self):
        """Test if pixels to indices mapping is correctly set up."""
        expected_mapping = {0: 0, 1: 1, 2: 2}
        self.assertEqual(self.class_mapping.pixels_to_indices_mapping, expected_mapping)

    def test_background_class_handling(self):
        """Test if the background class is correctly identified and handled."""
        self.assertEqual(self.class_mapping.background_class["label"], "background")
        self.assertEqual(self.class_mapping.background_class["pixel_value"], 0)

    def test_invalid_class_definitions(self):
        """Test if initializing with invalid class definitions raises an exception."""
        invalid_class_definitions = [
            {"label": "class1", "pixel_value": 1},
            {"label": "class2", "pixel_value": 2},
        ]  # Missing background class
        with self.assertRaises(Warning):
            ClassMapping(invalid_class_definitions)


if __name__ == '__main__':
    unittest.main()
