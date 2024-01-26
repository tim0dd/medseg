import click
import torch
from beartype import beartype
from medseg.verification.models.profiler.profiler import get_model_profile

from medseg.config.config import load_and_parse_config
from medseg.models.model_builder import build_model
from medseg.util.path_builder import PathBuilder


@click.command()
@click.option('--path', type=str, help='Path to config file')
@beartype
def calculate_flops(path: str):
    torch.hub.set_dir(PathBuilder.pretrained_dir_builder().build())
    cfg = load_and_parse_config(path)
    model = build_model(cfg, out_channels=1)
    spatial_dim = cfg["architecture"]["in_size"]
    inputs = torch.randn(1, 3, spatial_dim, spatial_dim)
    try:
        print("Trying to calculate MACS using torchprofile...")
        from torchprofile import profile_macs
        macs = profile_macs(model, inputs)
        print(f"MACS for the model (torchprofile): {macs}")
    except Exception as e:
        print(f"torchprofile exception: {e}")

    try:
        print("Trying to calculate GFLOPs using fvcore...")
        from fvcore.nn import flop_count
        flops_dict, unsupported_ops_dict = flop_count(model, (inputs,))
        total_flops = sum(flops_dict.values())
        print(f"GFLOPs for the model (fvcore): {total_flops}")
        print(f"Unsupported ops(fvcore): {unsupported_ops_dict}")
    except Exception as e:
        print(f"fvcore exception: {e}")

    try:
        print("Trying to calculate GFLOPs using torch profiler...")
        from medseg.verification.models.profiler.profiler import TIDSProfiler
        prof = TIDSProfiler(model)
        prof.start_profile()
        model(inputs)
        profile = prof.generate_profile()
        print(profile)
        prof.end_profile()
        flops, macs, params = get_model_profile(model=model, input_shape=(1, 3, spatial_dim, spatial_dim))
        print(f"GFLOPs for the model (torch profiler): {flops}")
        print(f"MACS for the model (torch profiler): {macs}")
        print(f"Params for the model (torch profiler): {params}")

    except Exception as e:
        print(f"torch profiler exception: {e}")


if __name__ == "__main__":
    calculate_flops()
