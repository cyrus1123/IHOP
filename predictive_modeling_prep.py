# -*- coding: utf-8 -*-
from typing import List, Union

import json
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from typing import Tuple

from labeled_combined_test_data import generate

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(PROJECT_ROOT_DIR, "processed")
DATA_HDF_PATH = os.path.join(PROCESSED_PATH, "output.h5")

if not os.path.exists(DATA_HDF_PATH):
    print("Processed data HDF not found, generating...")
    generate(DATA_HDF_PATH, min_points_per_interval=100)
    print("Done generating processed data HDF")

MAXIMUM_PERCENT_NA_PER_ROW = 50
PCA_VARIANCE_PORTION = 0.95
N_FEATURES = 60

class FeatureSelector:
    def __init__(self, n_features):
        self.classifier = ExtraTreesClassifier()
        self.n_features = n_features
        self.x_train = self.y_train = None

    def fit(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
        self.x_train, self.y_train = x_train, y_train
        return self

    def transform(self, x_to_transform):
        feature_importances = pd.Series(self.classifier.feature_importances_)
        most_important_features = feature_importances\
            .sort_values(ascending=False)\
            .head(self.n_features)\
            .index
        x_transformed = x_to_transform[:, most_important_features]
        return x_transformed

def score(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")

def get_k_fold(n_splits: int = 5, n_repeats: int = 10) \
        -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

def get_most_important_features(
        x: pd.DataFrame, y: pd.Series, n_features: int = N_FEATURES) -> pd.Index:
    feature_importances = np.zeros(len(x.columns))
    k_fold = get_k_fold()
    for train_index, test_index in k_fold.split(x, y):
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        imputer = KNNImputer()
        scaler = StandardScaler()

        x_train_imputed = imputer.fit_transform(x_train)
        x_train_scaled = scaler.fit_transform(x_train_imputed)
        feature_selector = ExtraTreesClassifier()
        feature_selector.fit(x_train_scaled, y_train)
        feature_importances += feature_selector.feature_importances_

    feature_importances = pd.Series(feature_importances)
    most_important_features = feature_importances.sort_values(ascending=False)\
        .head(n_features).index
    return x.columns[most_important_features]

def scale(x_train: pd.DataFrame, *x_to_scale: pd.DataFrame) -> \
        Union[pd.DataFrame, List[pd.DataFrame]]:
    # TODO: Check if this causes any problems
    if len(x_to_scale) > 0 and \
            not all((all(x.columns == x_train.columns) for x in x_to_scale)):
        raise ValueError("All x must have the same features as x_train")
    scaler = StandardScaler()
    scaler.fit(x_train)

    ind = x_train.index
    cols = x_train.columns

    x_train_transformed = pd.DataFrame(scaler.transform(x_train), ind, cols)
    if len(x_to_scale) == 0:
        return x_train_transformed

    result = [x_train_transformed]
    for x in x_to_scale:
        result.append(pd.DataFrame(scaler.transform(x), x.index, cols))
    return result

def get_train_and_test(x: pd.DataFrame, y: pd.Series) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    k_fold = get_k_fold()
    train_index, test_index = next(k_fold.split(x, y))
    # x_train, x_test = scale(x.iloc[train_index], x.iloc[test_index])
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return x_train, x_test, y_train, y_test

def get_json(activity_type: str, performance: str, importance: str,
             appreciation: str) -> str:
    """
    Given the activity type, performance, importance, and appreciation, return
    a JSON string that represents all of these together.
    :param activity_type: The activity type
    :param performance: The performance
    :param importance: The importance
    :param appreciation: The appreciation
    :return: A JSON string
    """
    return json.dumps({
        "activityType": activity_type,
        "performance": performance,
        "importance": importance,
        "appreciation": appreciation
    })

def from_json(json_string: str):
    """
    Given a JSON string of encoded data as output by get_json, return the
    activity type, performance, importance, and appreciation in a tuple, in
    the order stated.
    :param json_string: The JSON string representing the data
    :return: (activity type, performance, importance, appreciation)
    """
    json_dict = json.loads(json_string)
    activity_type = json_dict["activityType"]
    performance = json_dict["performance"]
    importance = json_dict["importance"]
    appreciation = json_dict["appreciation"]
    return (activity_type, performance, importance, appreciation)

def clean(df: pd.DataFrame,
          max_percent_na_per_row: float = MAXIMUM_PERCENT_NA_PER_ROW,
          drop_arousal: bool = False,
          drop_questionnaire_answers: bool = True,
          drop_self_perceived_arousal: bool = True,
          self_perceived_arousal_dummies: bool = True) -> pd.DataFrame:
    result = df
    # Drop columns that are not to be used in any analysis
    irrelevant_columns = ["activityId"]
    if drop_arousal:
        irrelevant_columns += ["arousal", "arousal2"]
    if drop_questionnaire_answers:
        irrelevant_columns += ["activityType", "performance", "importance",
                              "appreciation"]
    # Noise data is bad, so we need to drop it
    bad_noise_columns = [col for col in df.columns
                         if col.strip() in ("decibels", "dBA")]
    to_drop = irrelevant_columns + bad_noise_columns
    result = result.drop(to_drop, axis=1)
    result = result.replace([-np.inf, np.inf], np.nan)

    # Drop all rows with NA output
    if not drop_arousal:
        result = result.loc[result["arousal"].notna()]
    if not drop_questionnaire_answers:
        result = result.loc[result[
            ["activityType", "performance", "importance", "appreciation"]
        ].notna().all(axis=1)]
    if not drop_self_perceived_arousal:
        if self_perceived_arousal_dummies:
            # Apply one-hot encoding to self-perceived arousal
            result = pd.concat(
                [pd.get_dummies(
                    result["self_perceived_arousal"], prefix="sparousal"),
                    result],
                axis=1)
            # The column is left behind but is not needed anymore, so drop it
            result = result.drop("self_perceived_arousal", axis=1)
    else:
        result = result.drop("self_perceived_arousal", axis=1)

    # Drop all rows where more than the allowed number of values is NA
    result["percent_na"] = result.isnull().mean(axis=1) * 100
    result = result.loc[result["percent_na"] <= max_percent_na_per_row]
    result = result.drop("percent_na", axis=1)

    # Apply one-hot encoding to user
    result = pd.concat(
        [pd.get_dummies(result["user"], prefix="user"), result], axis=1)
    result = result.drop("user", axis=1)


    # Drop all rows with duplicate indices
    result = result[~result.index.duplicated(keep="first")]

    return result


df = clean(pd.read_hdf(DATA_HDF_PATH), drop_arousal=False,
           drop_questionnaire_answers=True, drop_self_perceived_arousal=True)

# Make these available for data notebooks
x = df.drop(["arousal", "arousal2"], axis=1)
y1 = df["arousal"]
y2 = df["arousal2"]

df_no_arousal = clean(pd.read_hdf(DATA_HDF_PATH), drop_arousal=True,
                      drop_questionnaire_answers=False,
                      drop_self_perceived_arousal=True)

# Make these available for data notebooks
x_no_arousal = df_no_arousal.drop(["activityType", "performance", "importance",
                                   "appreciation"], axis=1)     # I don't think
                                                                # we actually
                                                                # need this
y = df_no_arousal.apply(lambda row: get_json(row["activityType"],
                                             row["performance"],
                                             row["importance"],
                                             row["appreciation"]),
             axis=1)

df_combined = clean(pd.read_hdf(DATA_HDF_PATH), drop_arousal=False,
                    drop_questionnaire_answers=False,
                    drop_self_perceived_arousal=True)
x_combined = df_combined.drop(["arousal", "arousal2", "activityType",
                               "performance", "importance", "appreciation"],
                              axis=1)
y1_combined = df_combined["arousal"]
y2_combined = df_combined["arousal2"]
y_questionnaire_combined = df_combined.apply(
    lambda row: get_json(row["activityType"], row["performance"],
                         row["importance"], row["appreciation"]), axis=1)
