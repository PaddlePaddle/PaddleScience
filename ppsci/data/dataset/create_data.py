import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))


import sys

sys.path.append("PaddleScience/examples/huo/ML_Pipeline/scripts/utils")
import os

"""
Authors: Dylan Jones and Nick Roberts
Emails: ddj123@uw.edu and nickrob320@gmail.com

This file creates a cleaned dataset based on the perovskite database. It is linked to a csv.
That contains data downloaded form the perovskite database.

All you have to do is toggle the desired columns at the top of the file (comment them out
or add more not found here) and the rest of the code will create a dataset based on those
desired columns.

Running this file in the console gives a detailed timeline of how the database is created.
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from pathlib import Path

FULL_PATH = Path.cwd() / "data" / "data" / "raw" / "photocell_database.csv"
print("Found datafile:", os.path.isfile(FULL_PATH))
DESIRED_COLUMNS = {
    "DEVICE_CHARACTERISTICS": ["Cell_area_measured"],
    "LAYERED_TRANSPORT": ["ETL_stack_sequence", "HTL_stack_sequence"],
    "LAYERED_SUBSRATE": ["Substrate_stack_sequence"],
    "LAYERED_PAIRED_BACK_CONTACT": [
        "Backcontact_stack_sequence",
        "Backcontact_thickness_list",
    ],
    "LAYERED_PAIRED_PEROVSKITE_COMPOSITION": [
        "Perovskite_composition_a_ions",
        "Perovskite_composition_a_ions_coefficients",
        "Perovskite_composition_b_ions",
        "Perovskite_composition_b_ions_coefficients",
        "Perovskite_composition_c_ions",
        "Perovskite_composition_c_ions_coefficients",
    ],
    "LAYERED_PEROVSKITE_ADDITIVE": ["Perovskite_additives_compounds"],
    "PEROVSKITE_STRUCTURE": [
        "Perovskite_single_crystal",
        "Perovskite_composition_perovskite_ABC3_structure",
        "Perovskite_composition_perovskite_inspired_structure",
    ],
    "LAYERED_BANDGAP": ["Perovskite_band_gap"],
    "PEROVSKITE_DEPOSITION": [
        "Perovskite_deposition_quenching_induced_crystallisation",
        "Perovskite_deposition_quenching_media",
        "Perovskite_deposition_solvent_annealing",
        "Perovskite_deposition_number_of_deposition_steps",
    ],
    "LAYERED_PAIRED_PEROVSKITE_DEPOSITION_SOLVENT": [
        "Perovskite_deposition_solvents",
        "Perovskite_deposition_solvents_mixing_ratios",
    ],
    "LAYERED_PAIRED_PEROVSKITE_THERMAL_ANNEALING": [
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
    ],
    "ELECTRICAL_PARAMETERS": ["JV_default_Jsc"],
}
TARGET_COLUMN = "JV_default_Jsc"
ALIASES = {
    "Perovskite_composition_a_ions": "a",
    "Perovskite_composition_b_ions": "b",
    "Perovskite_composition_c_ions": "c",
    "Perovskite_deposition_thermal_annealing_temperature": "temp",
    "Perovskite_deposition_thermal_annealing_time": "time",
    "Perovskite_band_gap": "bandgap",
    "Perovskite_deposition_solvents": "depo_solvent",
    "Perovskite_deposition_quenching_media": "quenching_media",
    "Substrate_stack_sequence": "substrate_stack",
    "Perovskite_additives_compounds": "additive",
    "Backcontact_stack_sequence": "backcontact",
}
NAN_ALIAS = {"Perovskite_additives_compounds": "Undoped"}
"""
Organize the columns into data categories (determines how their information is formatted in the
final csv)
"""
CATEGORICAL_COLUMNS = {
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Perovskite_single_crystal",
    "Perovskite_deposition_quenching_media",
    "Substrate_stack_sequence",
    "Perovskite_additives_compounds",
}
PAIRED_COLUMNS = []
NON_PAIRED_COLUMNS = []
LAYERED_COLUMNS = []
for k, v in DESIRED_COLUMNS.items():
    if "PAIRED" in k:
        PAIRED_COLUMNS.extend(v)
    else:
        NON_PAIRED_COLUMNS.extend(v)
    if "LAYERED" in k:
        LAYERED_COLUMNS.extend(v)
TOKEN_DATA_FORMATTING = {
    "Perovskite_deposition_solvents": {"return_nan_for_error": True},
    "Perovskite_deposition_solvents_mixing_ratios": {
        "return_nan_for_error": True,
        "normalize_coefficients": True,
    },
    "Perovskite_deposition_thermal_annealing_temperature": {"thermal_annealing": True},
    "Perovskite_deposition_thermal_annealing_time": {"thermal_annealing": True},
    "Backcontact_stack_sequence": {"return_nan_for_error": True},
    "Backcontact_thickness_list": {"return_nan_for_error": True},
    "Perovskite_composition_a_ions_coefficients": {"numeric": True},
    "Perovskite_composition_b_ions_coefficients": {"numeric": True},
    "Perovskite_composition_c_ions_coefficients": {"numeric": True},
}
TOKEN_CLEANER_EQUIVALENTS = {
    "EPA": "PEA",
    "PEI": "PEA",
    "PDA": "PEA",
    "Ca": "Cs",
    "IM": "IA",
    "BDA": "BEA",
    "GA": "GU",
    "x": "nan",
}
LAYERED_GROUPS = {}
NON_STANDARDIZED_COLUMNS = set()
NON_STANDARDIZED_COLUMNS.add(TARGET_COLUMN)
"""
Define the helper functions for the pipeline
"""


def modification_wrapper(process_func):
    """
    Wraps all functions that modify the dataframe - prints info on how it changes.
    Functions MUST had df as their first argument e.g. func(df, *args, **kwargs).
    """

    def call_func(df, *args, **kwargs):
        curr_len = len(df)
        print(f"Calling {process_func.__name__}")
        print(f"\tCurrent number of datapoints: {curr_len}")
        resulting_df = process_func(df, *args, **kwargs)
        new_len = len(resulting_df)
        drop_amount = curr_len - new_len
        percent_change = round(100 * (drop_amount / curr_len), 1)
        overall_percent_change = round(100 * (drop_amount / STARTING_LEN), 1)
        print(f"\tDropped {drop_amount} datapoints")
        print(
            f"\tLocal change: {percent_change}% Global change: {overall_percent_change}%"
        )
        return resulting_df

    return call_func


@modification_wrapper
def drop_nan_in_target(df):
    """
    Drops all NaN rows in the target column.
    """
    df.dropna(how="all", subset=[TARGET_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@modification_wrapper
def drop_non_abx3(df):
    """
    Drops all non-abx3 structures from df
    """
    df = df[df["Perovskite_composition_perovskite_ABC3_structure"] == True]
    df = df[df["Perovskite_composition_perovskite_inspired_structure"] == False]
    return df


def normalize_data_token(token: str, token_cleaner: dict, thermal_annealing=False):
    """
    Takes a data token and normalizes it. It will convert numbers to floats;
    If thermal annealing is set to True, Unknown strings are converted to nan strings.
    """
    token = token.replace("(", "").replace(")", "")
    if token in token_cleaner:
        return token_cleaner[token]
    try:
        if token != "In":
            token = eval(token)
    except:
        if thermal_annealing:
            if token == "Unknown":
                token = "nan"
        pass
    return token


def split_tokens_by_layer(
    tokens: str,
    token_cleaner: dict = TOKEN_CLEANER_EQUIVALENTS,
    normalize_token=normalize_data_token,
    normalize_coefficients=False,
    return_nan_for_error=False,
    thermal_annealing=False,
    paired_layer=None,
    numeric=False,
):
    """
    Takes a string of tokens and removes parenthesis, replaces
    equivalent chemicals with a standard form, and splits them into
    a list of layers. If 'nan' in tokens, will flag as FALSE for
    include, unless return_nan_for_error is True.

    normalize_token is the function that normalize each token

    token_cleaner is a dictionary with equivalent forms

    normalize_coefficients = True | False -> determines if all tokens must sum to 1 to be included

    thermal_annealing = True | False -> handles how "Unknown" values are handled in normalize_token

    numeric = True | False -> determines if all tokens must be numeric to be included
    """
    include = True
    tokens = str(tokens)
    if "nan" in tokens and not return_nan_for_error:
        return False, "nan string: " + tokens
    layered_tokens = []
    tokens = tokens.replace(" >> ", "; ")
    layered_split = re.split("[\\s]*[|][\\s]*", tokens)
    if paired_layer and "nan" in tokens and return_nan_for_error:
        nan_array = []
        for layer in paired_layer:
            nan_array.append(["nan"] * len(layer))
        return True, nan_array
    for layer in layered_split:
        tokens = re.split("[\\s]*[;][\\s]*", layer)
        tokens = [
            normalize_token(x, token_cleaner, thermal_annealing=thermal_annealing)
            for x in tokens
        ]
        if normalize_coefficients:
            try:
                total = sum(tokens)
                tokens = [round(token / total, 2) for token in tokens]
            except:
                include = False
                return include, "not normalizable: " + str(tokens)
        if numeric:
            try:
                total = sum(tokens)
            except:
                include = False
                return include, "not numeric: " + str(tokens)
        layered_tokens.append(tokens)
    return include, layered_tokens


def check_dimensions(list1, list2) -> bool:
    """
    Returns whether the dimesions of two 2D lists match.
    Useful to see if formatted layered data is input correctly.
    """
    try:
        for i, layer in enumerate(list1):
            if len(layer) != len(list2[i]):
                return False
        return True
    except:
        return False


def split(token: str, header) -> tuple:
    if header in TOKEN_DATA_FORMATTING:
        col_kwargs = TOKEN_DATA_FORMATTING[header]
        include, layered_col = split_tokens_by_layer(token, **col_kwargs)
    else:
        include, layered_col = split_tokens_by_layer(token)
    return include, layered_col


def extract_paired_information(
    row_info: dict, row: pd.Series, columns: set, split_f=split
) -> list:
    """
    Takes a dictionary rowinfo that stores all the information for a given pandas row
    and extracts all the paired column information stored in the pandas series row and
    gives set the label as the key and the data as the value in row_info. It checks if
    the data is in a valid format as well, which it returns as the booelean include.
    """
    include = True
    for i in range(0, len(PAIRED_COLUMNS), 2):
        header = PAIRED_COLUMNS[i]
        header_value = row[PAIRED_COLUMNS[i]]
        data_header = PAIRED_COLUMNS[i + 1]
        data_header_value = row[PAIRED_COLUMNS[i + 1]]
        include_labelsi, layered_labels = split_f(header_value, header)
        include_datai, layered_data = split_f(data_header_value, data_header)
        include = (
            include
            and include_labelsi
            and include_datai
            and check_dimensions(layered_labels, layered_data)
        )
        if include:
            for j, layer in enumerate(layered_labels):
                for k, label in enumerate(layer):
                    prefactor = header
                    if PAIRED_COLUMNS[i] in ALIASES:
                        prefactor = ALIASES[PAIRED_COLUMNS[i]] + "_"
                    keyjk = prefactor + str(label) + f"_L{j}"
                    columns.add(keyjk)
                    row_info[keyjk] = layered_data[j][k]
    return row_info, include


def extract_information(row_info: dict, row: pd.Series, columns, split_f=split):
    """
    Takes a row_info dictionary and a pandas series of a dataframe and populates the
    row_info dictionary with datapoints of non-paired columns.

    Loops over non-paired columns and first determine if they are layered or not.
    If they are layered, create the appropriate layered headings and extract data.
    If not, just extract the data.
    """
    include = True
    for header in NON_PAIRED_COLUMNS:
        if header in LAYERED_COLUMNS:
            include_datai, layered_datai = split_f(row[header], header)
            include = include_datai
            if include_datai:
                if header in ALIASES:
                    header = ALIASES[header]
                for j, layer in enumerate(layered_datai):
                    for k, data in enumerate(layer):
                        keyjk = header + f"_L{j}"
                        if header in LAYERED_GROUPS:
                            LAYERED_GROUPS[header].add(keyjk)
                        else:
                            LAYERED_GROUPS[header] = set([keyjk])
                        columns.add(keyjk)
                        row_info[keyjk] = data
        else:
            data = row[header]
            if header in ALIASES:
                header = ALIASES[header]
            row_info[header] = data
            columns.add(header)
    return row_info, include


def create_index_dictionary(df):
    """
    Creates an dictionary where each key corresponds to the index of the input dataframe,
    and the values are dictionaries with columns as keys and values as the datapoints
    """
    print("Creating intermediate dataframe")
    start_len = len(df)
    print("\tCurrent number of datapoints:", start_len)
    index_dic = {}
    columns = set()
    for index, row in df.iterrows():
        row_info = {}
        row_info, include_paired = extract_paired_information(
            row_info, row, columns=columns
        )
        row_info, include_non_paired = extract_information(
            row_info, row, columns=columns
        )
        include = include_non_paired and include_paired
        if include:
            index_dic[index] = row_info
    return index_dic, columns, start_len


def construct_dataframe(index_dict, columns, starting_length):
    """
    Constructs a dataframe given an dictionary of row data
    with each entry of the dictionary being another dictionary wity
    columns as keys and values as data.
    """
    row_array = []
    columns = list(columns)
    for row_index, row_dict in index_dict.items():
        row = [0] * len(columns)
        for header, data in row_dict.items():
            row[columns.index(header)] = data
        row_array.append(row)
    output_csv = pd.DataFrame(row_array, columns=columns)
    output_csv = output_csv.loc[:, (output_csv != 0).any(axis=0)]
    for s in LAYERED_GROUPS.values():
        for col in s.copy():
            if col not in output_csv.columns:
                s.remove(col)
    new_len = len(output_csv)
    drop_amount = starting_length - new_len
    percent_change = 100 * round(drop_amount / starting_length, 1)
    overall_percent_change = round(100 * (drop_amount / STARTING_LEN), 1)
    print(f"\tDropped {drop_amount} datapoints")
    print(f"\tLocal change: {percent_change}% Global change: {overall_percent_change}%")
    print("Made intermediate dataframe")
    return output_csv


def categorical_to_one_hot(
    df: pd.DataFrame, non_standard_cols=NON_STANDARDIZED_COLUMNS
) -> pd.DataFrame:
    """
    Takes a pandas dataframe and converts all categorical columns specified in
    CATEGORICAL COLUMNS into one-hot encodings. It will fill NaN values with whatever
    is specified in NaN alias (else it defaults to "unknown")
    """
    print("Converting categorical columns to one-hot")
    for group in CATEGORICAL_COLUMNS:
        if group in NAN_ALIAS:
            nan_alias = NAN_ALIAS[group]
        else:
            nan_alias = "Unknown"
        if group in ALIASES:
            group = ALIASES[group]
        if group in LAYERED_GROUPS:
            group_columns = list(LAYERED_GROUPS[group])
        else:
            group_columns = group
        group_df = pd.DataFrame(df[group_columns]).fillna(nan_alias)
        df = df.drop(columns=group_columns)
        group_df.replace(0, np.nan, inplace=True)
        one_hot_group_df = pd.get_dummies(group_df, dummy_na=False).astype(int)
        non_standard_cols = non_standard_cols | set(one_hot_group_df.columns)
        df = pd.concat([df, one_hot_group_df], axis=1)
    return df, non_standard_cols


def remove_more_than_3std_away(
    df: pd.DataFrame, non_standard_cols: set
) -> pd.DataFrame:
    """
    Removes all rows with datapoints that are more than 3 standard deviations away
    from the column's mean.
    """
    l = len(df)
    drop_indices = set()
    for head, series in df.items():
        if series.name not in non_standard_cols:
            series = series.astype(np.float64)
            non_zero_data = series[series != 0]
            mean = non_zero_data.mean()
            std = non_zero_data.std()
            threshold = 3 * std
            outliers = non_zero_data[abs(non_zero_data - mean) > threshold]
            drop_indices.update(set(outliers.index))
    df = df.drop(list(drop_indices))
    df = df.loc[:, (df != 0).any(axis=0)]
    non_standard_cols = non_standard_cols.intersection(df.columns)
    return df, non_standard_cols


@modification_wrapper
def drop_empty_cols(df: pd.DataFrame) -> pd.DataFrame:
    l = len(df.columns)
    columns_to_drop = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=columns_to_drop)
    print(f"\tDropped {l - len(df.columns)} empty columns")
    return df


@modification_wrapper
def remove_duplicates(df: pd.DataFrame):
    """
    Removes all duplicate rows from the dataframe excluding the target column.
    Duplicate groups of rows with multiple values for the target are replaced with the mean of the
    target, or the mode of the target (whichever is greater) for the group of duplicate rows.
    """
    print("\tNon-NaN length before removal:", len(df.dropna()))
    target_col = df.pop(TARGET_COLUMN)
    df = pd.concat([df, target_col], axis=1)
    df = df.fillna("nan")
    group_cols = list(df.columns)
    group_cols.remove(TARGET_COLUMN)
    group_cols.append(TARGET_COLUMN)
    df = df[group_cols]
    group_cols.remove(TARGET_COLUMN)
    grouped = df.groupby(group_cols)
    no_dupes_array = []
    for key in grouped.groups:
        current = grouped.get_group(key)
        if len(current) > 1:
            key = list(key)
            jsc_series = current[TARGET_COLUMN]
            mean = jsc_series.mean()
            median = jsc_series.median()
            key.append(np.mean([mean, median]))
            no_dupes_array.append(key)
        else:
            no_dupes_array.append(list(current.to_numpy().squeeze()))
    no_dupes = pd.DataFrame(no_dupes_array, columns=df.columns)
    no_dupes.replace("nan", np.nan, inplace=True)
    print("\t Non-NaN length after removal:", len(no_dupes.dropna()))
    return no_dupes


@modification_wrapper
def standard_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard scaling on all features except those given in non_standard_cols
    """
    scaler = StandardScaler()
    non_standard_cols = TARGET_COLUMN
    non_scaled = df[non_standard_cols]
    df.drop(columns=non_standard_cols, inplace=True)
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df = pd.concat([df_scaled, non_scaled], axis=1)
    return df


def summarize_final_df(df: pd.DataFrame):
    """
    Summarizes the properties of the final dataframe
    """
    print("Pipeline concluded")
    print(f"\tFinal datapoints: {len(df)}")
    print(f"\tNumber of columns: {len(df.columns)}")
    print(f"\tNumber rows with NaN: {len(df) - len(df.dropna())}")


def main():
    """Main Pipeline"""
    nan_flag = ["Unknown"]
    df = pd.read_csv(FULL_PATH, header=0, na_values=nan_flag, low_memory=False)
    global STARTING_LEN
    STARTING_LEN = len(df)
    print(f"Starting data cleaning pipeline for {TARGET_COLUMN} as the target.")
    print(f"Current number of datapoints: {STARTING_LEN}")
    desired_cols_list = []
    for group_list in DESIRED_COLUMNS.values():
        desired_cols_list.extend(group_list)
    df = df[desired_cols_list]
    df = drop_nan_in_target(df)
    df = drop_non_abx3(df)
    index_dict, columns, start_len = create_index_dictionary(df)
    output_csv = construct_dataframe(index_dict, columns, start_len)
    output_csv, categorical_and_target_cols = categorical_to_one_hot(output_csv)
    output_csv, categorial_and_target_cols = remove_more_than_3std_away(
        output_csv, categorical_and_target_cols
    )
    output_csv = output_csv[output_csv["JV_default_Jsc"] <= 50]
    output_csv = remove_duplicates(output_csv)
    output_csv = drop_empty_cols(output_csv)
    output_csv = standard_scale_features(output_csv)
    output_path = Path.cwd() / "data" / "data" / "cleaned" / "cleaned_data.csv"
    output_csv.to_csv(output_path, index=False)
    summarize_final_df(output_csv)


if __name__ == "__main__":
    main()
