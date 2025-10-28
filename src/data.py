import pandas as pd 
import pandas as pd
import requests
from io import StringIO

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


if __name__ == "__main__":
    df = read_local_raw_data()
    if df is not None:
        print(df.head())

