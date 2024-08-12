import os

"""
Authors: Dylan Jones and Nick Roberts
Emails: ddj123@uw.edu and nickrob320@gmail.com

This file splits cleaned data into the training, validation, and testing sets.
These splits can be adjusted by editing DATASPLIT, as well as the randomized seed
replicatable results.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASPLIT = [0.8, 0.1, 0.1]
SEED = 42
INPUT_PATH = Path.cwd() / "data" / "data" / "cleaned" / "cleaned_data.csv"
OUTPUT_PATH = Path.cwd() / "data" / "data" / "cleaned"
if sum(DATASPLIT) != 1:
    raise ValueError("Invalid datasplit, must sum to 1.")
file_found = os.path.isfile(INPUT_PATH)
print("Found datafile:", file_found)
if not file_found:
    raise ValueError(
        "Cleaned dataset not found. Make sure to run create_data.py before spliting data."
    )
print("\tLoading Data... ", end="")
X = pd.read_csv(INPUT_PATH)
print("Complete")
l = len(X)
print(f"\t{l} entries")
target = input("Target column name: ")
if target not in X.columns:
    raise ValueError("Target column not found.")
drop_char = input("Remove NaN entries? (y, n): ")
if drop_char.lower() == "y":
    drop = True
elif drop_char.lower() == "n":
    drop = False
else:
    print(f"{drop_char} not a valid input, defaulting to removing NaN entries.")
    drop = True
if drop:
    print("Dropping NaN entries...")
    X.dropna(inplace=True)
    print(f"\t{l - len(X)} entries removed")
    print(f"\t{len(X)} entries remaining")
    l = len(X.columns)
    columns_to_drop = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=columns_to_drop)
    print(f"\tDropped {l - len(X.columns)} empty columns due to NaN removal")
    print(f"\t{len(X.columns)} columns remaining")
print("Splitting datasets...")
y = X[target]
del X[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=DATASPLIT[1] + DATASPLIT[2], random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=DATASPLIT[2] / (DATASPLIT[1] + DATASPLIT[2]),
    random_state=SEED,
)
X_train.to_csv(os.path.join(OUTPUT_PATH, "training.csv"), index=False, header=True)
y_train.to_csv(
    os.path.join(OUTPUT_PATH, "training_labels.csv"), index=False, header=True
)
X_val.to_csv(os.path.join(OUTPUT_PATH, "validation.csv"), index=False, header=True)
y_val.to_csv(
    os.path.join(OUTPUT_PATH, "validation_labels.csv"), index=False, header=True
)
X_test.to_csv(os.path.join(OUTPUT_PATH, "test.csv"), index=False, header=True)
y_test.to_csv(os.path.join(OUTPUT_PATH, "test_labels.csv"), index=False, header=True)
print("**COMPLETE**")
print(f"\tTraining set: {len(X_train)} entries")
print(f"\tValidation set: {len(X_val)} entries")
print(f"\tTest set: {len(X_test)} entries")
