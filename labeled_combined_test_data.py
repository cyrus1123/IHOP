from datetime import datetime
import os
from os.path import abspath, dirname, join
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import List, Optional, Union, Any
import warnings

from raw_data_constants import OUTPUT_PATH as RAW_DATA_PATH
from raw_data_hdf_generator import generate_hdf

TimeInterval = Union[str, pd.Timedelta]

PROJECT_ROOT_DIR = dirname(abspath(__file__))  # Directory containing this file
PROCESSED_PATH = join(PROJECT_ROOT_DIR, "processed")
OUTPUT_PATH = join(PROCESSED_PATH, "output.h5")

if not os.path.exists(RAW_DATA_PATH):
    print("Raw data HDF not found, generating...")
    generate_hdf()
    print("Done generating raw data HDF")

try:
    os.mkdir(PROCESSED_PATH)
except FileExistsError:
    # The fact that the directory we are trying to make already exists is a
    # serious problem that can cause severe and catastrophic system failure
    # that will cause every nuclear weapon in the world to detonate
    pass

accelerometer, act_inst, markers, environment, gyroscope, heart_rate, \
noise_level, questionnaire, skin_temperature, eda, self_perceived_arousal = (
    pd.read_hdf(RAW_DATA_PATH, key) for key in (
    "accelerometer", "activity_instances", "markers", "environment",
    "gyroscope", "heart_rate", "noise_level", "questionnaire",
    "skin_temperature", "eda", "arousal_questionnaire")
)

# region Data preprocessing
skin_temperature.rename(columns={"value": "skin_temp"}, inplace=True)
environment.rename(columns={"temperature": "env_temp"}, inplace=True)
accelerometer["accel_magnitude"] = np.sqrt(accelerometer["avgX"]**2 +
                                           accelerometer["avgY"]**2 +
                                           accelerometer["avgZ"]**2)
gyroscope["gyr_magnitude"] = np.sqrt(gyroscope["avgX"]**2 +
                                     gyroscope["avgY"]**2 +
                                     gyroscope["avgZ"]**2)

def get_datetime(date: str, time: Any) -> Any:
    if pd.isna(time):
        return np.nan
    datetime_string = " ".join((date, time))
    return pd.to_datetime(datetime_string)

def get_subset(df: pd.DataFrame, activity_id: int) -> pd.DataFrame:
    return df.loc[df["activityId"] == activity_id]

def clean(df: pd.DataFrame, relevant_colnames: List[str]) -> pd.DataFrame:
    df = df.copy()
    for colname in relevant_colnames:
        df[colname] = medfilt(df[colname])
        df[colname] = lowess(df[colname], df.index, 0.01)
    return df

# Strip whitespace around markers, make their names somewhat prettier, and turn
# them into valid Python identifiers to make things easier
markers["marker"] = markers["marker"].map(
    lambda s: s.lower().strip().replace("/", "_").replace(" ", "_"))

# Turn all unique marker names into dummy columns of format marker__marker_name
markers = \
    pd.concat((markers, pd.get_dummies(markers["marker"], "marker_")), axis=1)

# Some of the markers have duplicated times, drop them
markers["time_index"] = markers.index
markers.drop_duplicates(keep="first", inplace=True)
markers.drop(["time_index"], axis=1, inplace=True)
# endregion

# region Relevant features and statistics
relevant_dfs_and_colnames = (
    (heart_rate, "heartRate"),
    (heart_rate, "vo2"),
    (heart_rate, "signalQuality"),
    (skin_temperature, "skin_temp"),
    (environment, "env_temp"),
    (environment, "humidity"),
    (environment, "atmosphericPressure"),
    (accelerometer, "accel_magnitude"),
    (gyroscope, "gyr_magnitude"),
    (noise_level, "decibels"),
    (noise_level, "dBA"),
    (noise_level, "averageDecibels"),
    (noise_level, "peakDecibels"),
    (eda, "resistance")
)

len_without_na = lambda arr: sum(map(lambda x: pd.notna(x), arr))
stat_functions = [
    ("num_points", len_without_na),    # Number of data points
    ("avg", np.nanmean),    # Mean after removing NA values
    ("min", np.min),        # Minimum value
    ("max", np.max),        # Maximum value
    ("st_dev", np.nanstd)   # Standard deviation after removing NA values
]
# endregion

# region functions
def get_relevant_colnames(df):
    tuples_subset = [tup for tup in relevant_dfs_and_colnames if tup[0] is df]
    return [colname for _, colname in tuples_subset]

def merge(*dfs: pd.DataFrame) -> pd.DataFrame:
    if len(dfs) <= 1:
        return dfs[0]
    return dfs[0].merge(merge(*dfs[1:]), how="outer", left_index=True,
                        right_index=True)

def performance_map(performance: str) -> bool:
    return performance not in ("very_poorly", "poorly", "average")

def importance_map(importance: str) -> bool:
    return importance not in ("waste_of_time", "rather_not", "no_interest")

def appreciation_map(appreciation: str) -> bool:
    return appreciation not in ("hate", "dislike", "dont_care")

def user16_arousal_rule(performance: str, importance: str, appreciation: str):
    performance = performance_map(performance)
    importance = importance_map(importance)
    appreciation = appreciation_map(appreciation)
    if performance and importance and appreciation:
        return "high"
    if performance and not importance and not appreciation:
        return "normal"
    if performance and not importance and appreciation:
        return "high"
    return np.nan

def user16_arousal_rule_2(performance: str, importance: str, appreciation: str):
    performance = performance_map(performance)
    importance = importance_map(importance)
    appreciation = appreciation_map(appreciation)
    if performance and importance and appreciation:
        return "high"
    if performance and not importance and not appreciation:
        return "normal"
    if performance and not importance and appreciation:
        return "normal"
    return np.nan

def user21_arousal(performance: str, importance: str, appreciation: str):
    performance = performance_map(performance)
    importance = importance_map(importance)
    appreciation = appreciation_map(appreciation)
    if performance and importance and appreciation:
        return "high"
    if not performance and importance and not appreciation:
        return "normal"
    if performance and importance and not appreciation:
        return "low"
    return np.nan

def user21_arousal_2(performance: str, importance: str, appreciation: str):
    performance = performance_map(performance)
    importance = importance_map(importance)
    appreciation = appreciation_map(appreciation)
    if performance and importance and appreciation:
        return "high"
    if not performance and importance and not appreciation:
        return "normal"
    if performance and importance and not appreciation:
        return "normal"
    return np.nan

def get_arousal_for_user16(arousal_rule, act_type, performance, importance,
                           appreciation):
    if act_type == "rest":
        return "low"
    if act_type == "daily_task":
        base_arousal = arousal_rule(performance, importance, appreciation)
        return "normal" if base_arousal == "high" else base_arousal
    return arousal_rule(performance, importance, appreciation)

def get_arousal(row):
    # TODO: Perhaps better arousal calculation in the future?
    if row["user"] == "user21":
        return user21_arousal(row["performance"], row["importance"],
                              row["appreciation"])
    return get_arousal_for_user16(user16_arousal_rule, row["activityType"],
                                  row["performance"], row["importance"],
                                  row["appreciation"])

def get_arousal_2(row):
    # TODO: Perhaps better arousal calculation in the future?
    if row["user"] == "user21":
        return user21_arousal_2(row["performance"], row["importance"],
                                row["appreciation"])
    return get_arousal_for_user16(user16_arousal_rule_2, row["activityType"],
                                  row["performance"], row["importance"],
                                  row["appreciation"])

def get_time_interval_stats_at_point(index: datetime,
                                     before: Optional[TimeInterval],
                                     after: Optional[TimeInterval],
                                     df: pd.DataFrame,
                                     colname: str,
                                     min_points_per_interval: int) \
        -> pd.Series:
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")
    if before is not None: before = pd.Timedelta(before)
    if after is not None: after = pd.Timedelta(after)

    data = df[colname]

    start_time = index - before if before is not None else index
    end_time = index + after if after is not None else index
    begin_range = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_range = end_time.strftime("%Y-%m-%d %H:%M:%S")
    data_points = data[begin_range:end_range].dropna()

    if len(data_points) <= min_points_per_interval: return pd.Series(name=index)

    names = []
    values = []
    for prefix, stat_func in stat_functions:
        names.append(f"{prefix}_{colname}")
        values.append(stat_func(data_points))
    result = pd.Series(values, index=names, name=index)
    return result

def get_time_interval_stats(indices: pd.DatetimeIndex,
                            before: Optional[TimeInterval],
                            after: Optional[TimeInterval], df: pd.DataFrame,
                            colname: str, min_points_per_interval: int) \
        -> pd.DataFrame:
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")

    result = pd.DataFrame()
    for i in range(len(indices)):
        time = indices[i]
        entry = get_time_interval_stats_at_point(time, before, after, df,
                                                 colname,
                                                 min_points_per_interval)
        result = result.append(entry)

    return result

def count_markers(indices: pd.DatetimeIndex, before: Optional[TimeInterval],
                  after: Optional[TimeInterval], marker_df: pd.DataFrame):
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")
    if before is not None: before = pd.Timedelta(before)
    if after is not None: after = pd.Timedelta(after)

    marker_locations = marker_df.loc[
                       :, marker_df.columns.str.startswith("marker__")].copy()

    columns = ["count_" + col for col in marker_locations.columns]
    result_data_frame = pd.DataFrame(columns=columns)

    for i in range(len(indices)):
        time = indices[i]
        begin = time - before if before is not None else time
        end = time + after if after is not None else time
        begin_range = begin.strftime('%Y-%m-%d %H:%M:%S')
        end_range = end.strftime('%Y-%m-%d %H:%M:%S')

        marker_points = marker_locations[begin_range:end_range]

        entry_dict = {}
        for col in marker_locations.columns:
            entry_dict["count_" + col] = np.sum(marker_points[col])
        entry = pd.Series(entry_dict, name=time)
        result_data_frame = result_data_frame.append(entry)

    return result_data_frame

def get_self_perceived_arousal(indices: pd.DatetimeIndex,
                               before: Optional[TimeInterval],
                               after: Optional[TimeInterval],
                               sp_arousal_series: pd.Series):
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")
    if before is not None: before = pd.Timedelta(before)
    if after is not None: after = pd.Timedelta(after)

    result_data_frame = pd.DataFrame(columns=["self_perceived_arousal"])

    for i in range(len(indices)):
        time = indices[i]
        begin = time - before if before is not None else time
        end = time + after if after is not None else time
        begin_range = begin.strftime('%Y-%m-%d %H:%M:%S')
        end_range = end.strftime('%Y-%m-%d %H:%M:%S')

        self_perceived_arousals = sp_arousal_series[begin_range:end_range]
        if len(self_perceived_arousals) > 0:
            self_perceived_arousal = self_perceived_arousals.iloc[-1]
        else:
            self_perceived_arousal = pd.NA

        entry = pd.Series({"self_perceived_arousal": self_perceived_arousal},
                          name=time)
        result_data_frame = result_data_frame.append(entry)

    return result_data_frame

def get_stats_for_activity(activity_id: int, before: Optional[TimeInterval],
                           after: Optional[TimeInterval],
                           min_points_per_interval: int,
                           use_self_perceived_arousal_index: bool = False) \
        -> pd.DataFrame:
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")

    cols = []
    if use_self_perceived_arousal_index:
        spa_data = get_subset(self_perceived_arousal, activity_id)\
            [["source", "activityId", "selfPerceivedArousal"]]\
            .rename(columns={"source": "user"})
        time_index = spa_data.index
        cols.append(spa_data)
    else:
        ques_data = get_subset(questionnaire, activity_id)\
            .drop(["activity", "event", "version", "time"], axis=1)\
            .rename(columns={"source": "user"})
        time_index = ques_data.index.drop_duplicates()
        cols.append(ques_data)
    def append_to_list(df, colname):
        df_ss = get_subset(df, activity_id)
        df_ss = clean(df_ss, get_relevant_colnames(df_ss))
        stats = get_time_interval_stats(time_index, before, after, df_ss,
                                        colname, min_points_per_interval)
        cols.append(stats)

    for df, colname in relevant_dfs_and_colnames:
        if colname == "vo2" and activity_id <= 122:
            continue    # Activities 122 and before have bad VO2 data
        append_to_list(df, colname)

    marker_ss = get_subset(markers, activity_id)
    marker_count_df = count_markers(time_index, before, after, marker_ss)
    cols.append(marker_count_df)

    if not use_self_perceived_arousal_index:
        sp_arousal_ss = get_subset(self_perceived_arousal, activity_id)
        sp_arousal_df = \
            get_self_perceived_arousal(time_index, before, after,
                                       sp_arousal_ss["selfPerceivedArousal"])
        cols.append(sp_arousal_df)

    df = merge(*cols)
    df.index.name = "time"

    if not use_self_perceived_arousal_index and len(df.index) > 0:
        df["arousal"] = df.apply(get_arousal, axis=1)
        df["arousal2"] = df.apply(get_arousal_2, axis=1)

    return df
# endregion

def generate(output_path: Optional[str] = None,
             before: Optional[TimeInterval] = "30 minutes",
             after: Optional[TimeInterval] = None,
             min_points_per_interval: int = 0, return_result: bool = True,
             use_self_perceived_arousal_index: bool = False) \
        -> Optional[pd.DataFrame]:
    if before is None and after is None:
        raise ValueError("Must provide a value for at least one of "
                         "before, after")

    if output_path is None and not return_result:
        warnings.warn("Because no output path was specified and return_result "
                      "is false, this function call is useless and will do not"
                      "hing but waste computational resources. Please specify "
                      "an output path or set return_result to True when callin"
                      "g the generate function.")

    result_df = None
    for activity_id in act_inst.index:
        df_for_activity_id = \
            get_stats_for_activity(activity_id, before, after,
                                   min_points_per_interval,
                                   use_self_perceived_arousal_index)
        if result_df is None:
            result_df = df_for_activity_id
        else:
            result_df = result_df.append(df_for_activity_id)

    # Get rid of all rows that are exact duplicates
    result_df["time"] = result_df.index
    try:
        result_df.drop_duplicates(inplace=True)
    except TypeError:
        from IPython.display import display
        display(result_df)
        raise
    result_df.drop(["time"], axis=1, inplace=True)

    # Save to file
    if output_path is not None:
        result_df.to_hdf(output_path, "data")

    if return_result:
        return result_df

def generate_multiple_intervals(output_path: str,
                                min_points_per_interval: int = 0,
                                return_result: bool = True,
                                generate_even_if_exists: bool = True) \
        -> Optional[pd.DataFrame]:
    if not generate_even_if_exists and os.path.exists(output_path):
        if return_result:
            return pd.read_hdf(output_path)
        return

    df_30min, df_20min, df_15min, df_10min, df_5min, df_1min, df_30s = (
        generate(before=time, min_points_per_interval=min_points_per_interval)
        for time in (
        "30 minutes",
        "20 minutes",
        "15 minutes",
        "10 minutes",
        "5 minutes",
        "1 minute",
        "30 seconds"
    )
    )
    non_time_columns = ["user", "dataset", "activityId", "activityType",
                        "performance", "importance", "appreciation", "arousal",
                        "arousal2"]
    dfs = (df_30min, df_20min, df_15min, df_10min, df_5min, df_1min, df_30s)
    suffixes = ("30_min", "20_min", "15_min", "10_min", "5_min", "1_min",
                "30_sec")
    for df, suffix in zip(dfs, suffixes):
        df.rename({col: f"{col}_{suffix}" for col in df.columns
                   if col not in non_time_columns}, axis=1, inplace=True)
        if df is not df_30min:
            df.drop(non_time_columns, axis=1, inplace=True)

    result = pd.concat(dfs, axis=1)
    result.to_hdf(output_path, "data")
    if return_result:
        return result