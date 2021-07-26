import os
import pandas as pd
from typing import AnyStr

from raw_data_constants import ALL_CSV_PATHS, QUESTIONNAIRE_PATH

def assert_all_files_exist() -> None:
    """
    Raise a FileNotFoundError if not all the required CSV files exist.
    :return: None
    """
    nonexistent_files = []
    for path in ALL_CSV_PATHS:
        if not os.path.exists(path):
            nonexistent_files.append(path)
    if len(nonexistent_files) > 0:
        raise FileNotFoundError(f"The following required files do not exist: "
                                f"{', '.join(nonexistent_files)}")


def read_csv(csv_path: AnyStr) -> pd.DataFrame:
    """
    Helper function to more easily read a CSV file into a DataFrame.
    :param csv_path: The path of the CSV
    :return: A DataFrame
    """
    if csv_path == QUESTIONNAIRE_PATH:
        index_column = 6
    else:
        index_column = 5

    df = pd.read_csv(csv_path, index_col=index_column, parse_dates=True)

    if csv_path == QUESTIONNAIRE_PATH:
        # The questionnaireTime in the questionnaire is formatted for UTC, so
        # need to convert our time (UTC-5h)
        df = df.tz_convert('US/Eastern')
        df = df.tz_localize(None)
        # Also, time is a string and needs to be converted to np.datetime64
        df["time"] = pd.to_datetime(df["time"])

    return df
