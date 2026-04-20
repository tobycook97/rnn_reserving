import pandas as pd 
import numpy as np
from collections import defaultdict


def read_data(file_path, data_debug=False):
    """Reads a CSV file and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if data_debug:
            return df[df['GRCODE'] == 43].copy()
        else:
            return df
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return None

def read_local_raw_data(data_debug=False):
    """Reads the CAS Actuarial Data from a predefined path."""
    file_path = '../data/raw/ppauto_pos.csv'
    return read_data(file_path, data_debug=data_debug)

def process_data(df_cas):
    """Processes the DataFrame"""

    df_cas['incurred_loss_ratio'] = (
        (df_cas['IncurLoss_B'] / df_cas['EarnedPremNet_B'])
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )

    df_cas['paid_loss_ratio'] = (
        (df_cas['CumPaidLoss_B'] / df_cas['EarnedPremNet_B'])
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )  # handles zero premium issues (like why - but we don't really care... )

    df_cas['case_loss_ratio'] = df_cas['incurred_loss_ratio'] - df_cas['paid_loss_ratio']

    df_cas['calendar_year'] = df_cas['AccidentYear'] + df_cas['DevelopmentLag'] - 1

    gr_code_mapping = {code: i for i, code in enumerate(df_cas['GRCODE'].unique())}
    # creates a mapping from GRCODE to integers.
    # this will allow us to embed the GRCODE as a categorical variable later on.
    df_cas['GRCODE_mapped'] = df_cas['GRCODE'].map(gr_code_mapping)
    
    return df_cas

def split_data(df_cas):
    """Splits the DataFrame into training, validation, 
    and test sets based on calendar year and development lag
    """
    conditions = [
        (df_cas["calendar_year"] <= 1995) & (df_cas["DevelopmentLag"] >= 1),
        (df_cas["calendar_year"] > 1995) & (df_cas["calendar_year"] <= 1997) & (df_cas["DevelopmentLag"] >= 1), # R guy did > 1 which was obbvs pretty interesting but unsure why...
        (df_cas["calendar_year"] > 1997)
    ]

    choices = ["train", "validation", "test"]

    df_cas["bucket"] = np.select(conditions, choices, default=None)
    return df_cas


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = 'train'
) -> tuple[list[np.ndarray], list[int], list[tuple[int, str]]]:
    """
    Group by AccidentYear and GRCODE_mapped, extract feature sequences for the given split.

    Returns:
        seqs: list of feature arrays
        lengths: list of sequence lengths
        ids: list of (AccidentYear, GRCODE_mapped) identifiers
    """
    input_seqs = []
    target_seqs = []
    lengths = []
    ids = []

    df_split = df[df['bucket'] == split].copy()

    for (ay, cc), group in df_split.groupby(['AccidentYear', 'GRCODE_mapped']):

        group = group.sort_values('DevelopmentLag')

        if len(group) < 2:
          continue

        seqs = group[feature_cols].values

        input_seq = seqs[:-1] # hide the last one!
        target_seq = seqs[1:, 0] # incurred loss ratio only

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

        lengths.append(len(input_seq))
        ids.append((ay, cc))

    return input_seqs, target_seqs, lengths, ids


def build_sequences(df, feature_cols):
    data = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }

    for (ay, cc), group in df.groupby(["AccidentYear", "GRCODE_mapped"]):

        group = group.sort_values("DevelopmentLag").reset_index(drop=True)
        
        for i in range(len(group) - 1):
            target_split = group.loc[i+1, "bucket"]
            
            input_seq = group.loc[:i, feature_cols].values
            
            future_mask = (
                (group.index > i) &
                (group["bucket"] == target_split)
            )
            
            target_seq = group.loc[future_mask, feature_cols[0]].values
            
            data[target_split]["inputs"].append(input_seq)
            data[target_split]["targets"].append(target_seq)
            data[target_split]["lengths"].append(i + 1)  # actual length before padding
            data[target_split]["ids"].append(
                (ay, cc, group.loc[i, "DevelopmentLag"], target_split)
            )

    return data

def read_and_process_data(
    feature_cols: list[str] = None,
    data_debug: bool = False,
):
    df_cas = read_local_raw_data(data_debug=data_debug)
    
    if df_cas is not None:
        df_cas = process_data(df_cas)
        df_cas = split_data(df_cas)
        sequence_data = build_sequences(df_cas, feature_cols)
        
        return sequence_data['train'], sequence_data['validation'], sequence_data['test']
    else:
        ValueError("Failed to read the raw data.")

