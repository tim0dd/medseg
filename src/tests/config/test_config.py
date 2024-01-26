import unittest

from medseg.config.config import merge_configs


class ConfigTest(unittest.TestCase):
    def test_merge_configs_simple_overwrite(self):
        priority_cfg = {"a": 1}
        other_cfg = {"a": 2}
        expected_result = {"a": 1}
        self.assertEqual(merge_configs(priority_cfg, other_cfg), expected_result)

    def test_merge_configs_nested_dict(self):
        priority_cfg = {"a": {"b": 1}}
        other_cfg = {"a": {"c": 2}}
        expected_result = {"a": {"b": 1, "c": 2}}
        self.assertEqual(merge_configs(priority_cfg, other_cfg), expected_result)

    def test_nested_list(self):
        priority_cfg = {"a": [1, 2, 3]}
        other_cfg = {"a": [4, 5]}
        expected_result = {"a": [1, 2, 3]}
        self.assertEqual(merge_configs(priority_cfg, other_cfg), expected_result)

    def test_merge_configs_nested_list_dict(self):
        priority_cfg = {"a": [{"b": 1}, {"c": 2}]}
        other_cfg = {"a": [{"b": 3}, {"d": 4}]}
        expected_result = {"a": [{"b": 1}, {"c": 2, "d": 4}]}
        self.assertEqual(merge_configs(priority_cfg, other_cfg), expected_result)

    def test_merge_configs_mixed_structure(self):
        priority_cfg = {"a": {"b": [{"c": 1}, {"d": 2}], "e": [1, 2, 3]}}
        other_cfg = {"a": {"b": [{"c": 3}, {"e": 4}], "e": [4, 5]}, "f": 6}
        expected_result = {
            "a": {"b": [{"c": 1}, {"d": 2, "e": 4}], "e": [1, 2, 3]},
            "f": 6,
        }
        self.assertEqual(merge_configs(priority_cfg, other_cfg), expected_result)


if __name__ == "__main__":
    unittest.main()
