import unittest

from hyperopt.pyll.base import Apply

from medseg.config.parse import parse_hyperopt_cfg


class ParseTest(unittest.TestCase):
    def create_hyperopt_cfg(self, cfg: dict) -> dict:
        """Helper method to wrap cfg in the nested structure."""
        return {'hyperopt': {'param_space': cfg}}

    def get_value(self, cfg: dict, key: str):
        """Helper method to get the value from the nested structure."""
        return cfg['hyperopt']['param_space'][key]

    def test_parse_hyperopt_cfg(self):
        cfg = self.create_hyperopt_cfg({
            "lr": [0.001, 0.1],
            'batch_size': [1, 2, 4, 8],
        })
        parsed_cfg = parse_hyperopt_cfg(cfg)
        self.assertIsInstance(self.get_value(parsed_cfg, 'lr'), Apply)
        self.assertIsInstance(self.get_value(parsed_cfg, 'batch_size'), Apply)

    def test_parse_hyperopt_cfg_list(self):
        cfg = self.create_hyperopt_cfg({'a': [0.001, 0.1], 'b': [1, 2, 4, 8]})
        parsed_cfg = parse_hyperopt_cfg(cfg)
        self.assertIsInstance(self.get_value(parsed_cfg, 'a'), Apply)
        self.assertIsInstance(self.get_value(parsed_cfg, 'b'), Apply)

    def test_parse_hyperopt_cfg_nested(self):
        cfg = self.create_hyperopt_cfg({
            'optimizer': {
                'type': ["adam", "sgd"],
                'lr': [0.001, 0.01, 0.1]
            },
            'batch_size': [1, 2, 4, 8],
        })
        parsed_cfg = parse_hyperopt_cfg(cfg)
        self.assertIsInstance(self.get_value(parsed_cfg, 'optimizer')['type'], Apply)
        self.assertIsInstance(self.get_value(parsed_cfg, 'optimizer')['lr'][0], Apply)
        self.assertIsInstance(self.get_value(parsed_cfg, 'optimizer')['lr'][1], Apply)
        self.assertIsInstance(self.get_value(parsed_cfg, 'batch_size'), Apply)

    def test_parse_hyperopt_cfg_no_parse(self):
        cfg = {
            'learning_rate': 0.001,
            'batch_size': 8,
        }
        parsed_cfg = parse_hyperopt_cfg(cfg)
        self.assertEqual(parsed_cfg, cfg)

    def test_parse_hyperopt_cfg_no_parse_mixed(self):
        cfg = self.create_hyperopt_cfg({
            'learning_rate': [0.001, 0.1],
            'batch_size': 8,
        })
        parsed_cfg = parse_hyperopt_cfg(cfg)
        self.assertIsInstance(self.get_value(parsed_cfg, 'learning_rate'), Apply)
        self.assertEqual(self.get_value(parsed_cfg, 'batch_size'), 8)
