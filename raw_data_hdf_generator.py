import numpy as np
import pandas as pd
from typing import Tuple

from raw_data_hdf_generator_functions import read_csv
from raw_data_constants import ACCELEROMETER_PATH, ACT_INST_PATH, \
    AROUSAL_QUESTIONNAIRE_PATH, EDA_PATH, ENVIRONMENT_PATH, GYROSCOPE_PATH, \
    HEART_RATE_PATH, MARKERS_PATH, NOISE_LEVEL_PATH, OUTPUT_PATH, \
    QUESTIONNAIRE_PATH, QUICK_NOTES_PATH, SKIN_TEMP_PATH


def _load_all_data() -> Tuple[pd.DataFrame]:
    act_inst = pd.read_csv(ACT_INST_PATH, index_col=0)

    (accelerometer, sp_arousal, markers, eda, environment, gyroscope,
     heart_rate, noise_level, questionnaire, quick_notes, skin_temperature) = \
        all_dfs = [read_csv(path) for path in (
            ACCELEROMETER_PATH, AROUSAL_QUESTIONNAIRE_PATH, MARKERS_PATH,
            EDA_PATH, ENVIRONMENT_PATH, GYROSCOPE_PATH, HEART_RATE_PATH,
            NOISE_LEVEL_PATH, QUESTIONNAIRE_PATH, QUICK_NOTES_PATH,
            SKIN_TEMP_PATH
        )]

    # In some data, some of the noise level values were taken as -infinity, fix
    # that problem
    noise_level.replace("-Infinity", -np.inf, inplace=True)
    noise_level.replace("Infinity", np.inf, inplace=True)
    # Also, for some reason, some values seem to be taken as strings; fix that
    noise_level["decibels"] = noise_level["decibels"].apply(
        lambda x: float(x) if isinstance(x, str) else x)
    noise_level["dBA"] = noise_level["dBA"].apply(
        lambda x: float(x) if isinstance(x, str) else x)

    # Get number of events and last event time
    def get_n_events(activity_id: int) -> int:
        result = 0
        for df in all_dfs:
            subset = df.loc[df["activityId"] == activity_id]
            result += len(subset)
        return result

    def get_last_event_time(activity_id: int):
        result = None
        for df in all_dfs:
            subset = df.loc[df["activityId"] == activity_id]
            if len(subset) == 0: continue
            last_event_time = subset.index[-1]
            if result is None or last_event_time > result:
                result = last_event_time
        return result

    for act_id in act_inst.index:
        act_inst.loc[act_id, "last_event"] = get_last_event_time(act_id)
        act_inst.loc[act_id, "num_events"] = get_n_events(act_id)

    # Now sort activity instance table in reverse order by ID
    act_inst.sort_values(["id"], ascending=False, inplace=True)

    # Fill missing values of end_time with last_event
    act_inst["end_time"].fillna(act_inst["last_event"], inplace=True)

    # Convert start_time and end_time to datetime
    act_inst["start_time"] = pd.to_datetime(act_inst["start_time"])
    act_inst["end_time"] = pd.to_datetime(act_inst["end_time"])

    # Add duration, which is the difference between end_time and start_time
    # (or last_event and start_time if last_event is later)
    true_end_time = act_inst[["end_time", "last_event"]].max(axis=1)
    from IPython.display import display
    display(true_end_time)
    display(act_inst)
    display(act_inst["start_time"])
    delta = true_end_time - act_inst.start_time
    act_inst["duration"] = delta

    # Convert number of events to integer
    act_inst["num_events"] = act_inst["num_events"].astype(int)

    return all_dfs + [act_inst]

def generate_hdf() -> None:
    all_dfs = _load_all_data()
    corresponding_keys = ("accelerometer", "arousal_questionnaire", "markers",
                          "eda", "environment", "gyroscope", "heart_rate",
                          "noise_level", "questionnaire", "quick_notes",
                          "skin_temperature", "activity_instances")
    for dataframe, key in zip(all_dfs, corresponding_keys):
        dataframe.to_hdf(OUTPUT_PATH, key)

if __name__ == "__main__":
    generate_hdf()