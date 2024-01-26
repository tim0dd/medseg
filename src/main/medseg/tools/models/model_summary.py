from typing import Tuple, Optional

import click
from beartype import beartype

from medseg.config.config import load_and_parse_config
from medseg.evaluation.params import create_model_summary
from medseg.models.model_builder import build_model


@click.command()
@click.option('--model_cfg', type=click.Path(exists=True), help='Path to model config file')
@click.option('--out_path', type=click.Path(), help='Path to save the results')
@click.option('--input_size', nargs=4, type=int, required=False, help='Input size as a tuple (B, C, H, W)')
@click.option('--out_channels', type=int, default=1, help='Number of output channels')
@beartype
def get_model_summary_from_cfg(model_cfg: str, out_path: str, input_size: Optional[Tuple], out_channels: int = 1):
    cfg = load_and_parse_config(model_cfg)
    model = build_model(cfg, out_channels=out_channels)
    if input_size is None:
        print("No input size specified. Trying to determine from given config.")
        img_size = cfg['architecture']['in_size']
        channels = 3
        input_size = (1, channels, img_size, img_size)
    print(create_model_summary(model, input_size, save_path=out_path))
    print(f"Saved model summary to: {out_path}")


if __name__ == '__main__':
    get_model_summary_from_cfg()
