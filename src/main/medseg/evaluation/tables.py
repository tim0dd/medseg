import json
import math
from typing import List, Dict, Optional

import click
import pandas as pd
from beartype import beartype
from pandas import Series
from scipy.stats import pearsonr


def scientific_notation(number: float) -> str:
    m, e = f'{number:.0e}'.split('e-')
    m, e = int(m), int(e)
    return fr"${m}\cdot10^{{-{e}}}$"


DEFAULT_HEADER_MAPPING = {
    "learnable_params": "Par.",
    "optimizer -> lr": "$\\alpha$",  # \a needs to be escaped
    "optimizer -> weight_decay": "$\\lambda$",
    "settings -> batch_size": "B",
    "val/final_dice": "mDSC",
    "val/final_iou": "mIoU",
    "val/final_precision": "mPrc",
    "val/final_recall": "mRec",
    "train/total_epochs": "Ep.",
    "architecture -> fcb_n_levels": "$F_{L}$",
    "architecture -> fcb_n_residual_block_per_level": "$R_{L}$",
    "architecture -> ph_rb_channels": "$PH_{C_{i}}$",
    "architecture -> unet_levels": "$U_{L}$",
    "architecture -> norm": "$Norm$",
    "architecture -> activation": "$Act$",
    "architecture -> use_ema": "$H_{EMA}$",
    "architecture -> deep_supervision": "$H_{DS}$",
    "architecture -> dropout": "$H_{Drop}$",
    "architecture -> encoder_drop_rate": "$S_{Enc_{Drop}}$",
    "architecture -> decoder_drop_rate": "$S_{Dec_{Drop}}$",
    "architecture -> mlp_embed_dim": "$S_{MLP}$",
    "architecture -> num_heads": "$S_{Heads}$",
    "architecture -> mscan_dropout_ratio": "$E_{Dr}$",
    "architecture -> ham_dropout_ratio": "$D_{Dr}$",
    "architecture -> ham_channels": "$D_{C}$",
    "architecture -> ham_align_channels": "$D_{AC}$",
    "architecture -> dec_channels": "$D_{C}$",
    "architecture -> drop_rate": "$E_{Dr}$",
    "architecture -> attn_drop_rate": "$E_{ADr}$",
    "architecture -> drop_path_rate": "$E_{DPR}$",
}

# DEFAULT_REORDERING = [0, 11, 4, 5, 6, 1, 2, 3, 8, 7, 9, 10]

DEFAULT_REORDERING = [0, 12, 5, 6, 7, 4, 3, 2, 1, 9, 8, 10, 11]


@beartype
def read_json_file(filename: str) -> Dict[str, List]:
    with open(filename) as file:
        data = json.load(file)
    return data


@beartype
def make_max_value_bold(s: Series) -> Series:
    is_max = s == s.max()
    s[is_max] = "$\\mathbf{" + s[is_max].astype(str) + "}$"
    return s


@beartype
def process_dataframe(data: Dict[str, List], model_abbreviation: str, n_rows: Optional[int] = None,
                      header_order: Optional[List[int]] = None,
                      header_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(data["rows"], columns=data["header"])

    if n_rows:
        df = df.iloc[:n_rows]
    metrics_cols = []
    for col in df.columns:
        if col == "optimizer -> lr":
            df[col] = df[col].apply(lambda x: scientific_notation(x))
        elif "val" in col or "test" in col:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}" if isinstance(x, (int, float)) else x)
            metrics_cols.append(col)
        elif col == "learnable_params":
            df[col] = df[col].apply(lambda x: f"{math.ceil(x / 1e6)}M")
        if df[col].dtype == float and all(df[col].apply(float.is_integer)):
            df[col] = df[col].apply(lambda x: int(x))
            print(col)

    df[metrics_cols] = df[metrics_cols].apply(make_max_value_bold, axis=0)

    df.insert(0, "M", pd.Series([f"${model_abbreviation}_{{{i}}}$" for i in range(1, len(df) + 1)]))

    header_order = header_order or DEFAULT_REORDERING
    if header_order:
        ordered_headers = ["M"] + [data["header"][i] for i in header_order]
        df = df[ordered_headers]

    if header_map:
        df.rename(columns=header_map, inplace=True)

    df.rename(columns=DEFAULT_HEADER_MAPPING, inplace=True)
    return df


@beartype
def convert_to_latex_table(df: pd.DataFrame, filename: str, longtable: bool) -> None:
    column_format = "r" * len(df.columns)
    latex_table = df.to_latex(index=False, escape=False, longtable=longtable, header=True, column_format=column_format,
                              caption='caption', label='label')
    # table = table.replace()

    with open(filename, 'w') as f:
        f.write(latex_table)



@click.group()
def tables():
    pass


@tables.command(name='results')
@click.option('--in_path', type=click.Path(exists=True))
@click.option('--out_path', type=click.Path())
@click.option('--model_abbreviation', '-m', default="X")
@click.option('--header_order', '-o', default=None, type=click.STRING)
@click.option('--header_map', '-h', default=None, type=click.STRING)
@click.option('--n_rows', '-n', default=None, type=click.INT)
@click.option('--longtable', '-l', is_flag=True)
@beartype
def create_results_table(in_path: str,
                         out_path: str,
                         model_abbreviation: str,
                         longtable: bool = False,
                         header_order: Optional[str] = None,
                         header_map: Optional[str] = None,
                         n_rows: Optional[int] = None) -> None:
    data = read_json_file(in_path)

    if header_order:
        header_order = list(map(int, header_order.split(',')))

    if header_map:
        header_map = json.loads(header_map)

    df = process_dataframe(data, model_abbreviation, n_rows, header_order, header_map)
    convert_to_latex_table(df, out_path, longtable)


@tables.command(name='correlations')
@click.option('--in_path', type=click.Path(exists=True), required=True)
@click.option('--out_path', type=click.Path(), required=True)
def create_correlations_table(in_path: str, out_path: str) -> None:
    with open(in_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['rows'], columns=data['header'])

    # calculate correlation coefficients and p-values for each of the hyperparameters
    correlations = []
    for param in data['header'][:-5]:  # Assuming last 5 columns are metric results
        r, p = pearsonr(df[param], df['val/final_dice'])
        correlations.append((param, r, p))

    correlations_df = pd.DataFrame(correlations, columns=['hyperparameter', 'correlation', 'p-value'])
    latex_table = correlations_df.to_latex(index=False)
    print(latex_table)
    with open(out_path, 'w') as f:
        f.write(latex_table)


if __name__ == "__main__":
    tables()
