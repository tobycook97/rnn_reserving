import pandas as pd 
import numpy as np

def read_data(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return None

def read_local_raw_data():
    """Reads the CAS Actuarial Data from a predefined path."""
    file_path = 'data/raw/ppauto_pos.csv'
    return read_data(file_path)

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

def read_and_process_data(
    feature_cols: list[str] = None
):
    df_cas = read_local_raw_data()
    if df_cas is not None:
        df_cas = process_data(df_cas)
        df_cas = split_data(df_cas)
        train_sequences = prepare_sequences(
            df_cas,
            feature_cols=feature_cols,
            split='train'
        )
        validation_sequences = prepare_sequences(
            df_cas,
            feature_cols=feature_cols,
            split='validation'
        )
        test_sequences = prepare_sequences(
            df_cas,
            feature_cols=feature_cols,
            split='test'
        )
        
    return train_sequences, validation_sequences, test_sequences


if __name__ == "__main__":

    feature_cols = [
        'paid_loss_ratio', 'DevelopmentLag', 'GRCODE_mapped'
    ]

    train_data = read_and_process_data(
        feature_cols=feature_cols,
    )

    val_data = read_and_process_data(
        feature_cols=feature_cols, 
    )

    if val_data is not None:
        print(val_data[0][0]) 


