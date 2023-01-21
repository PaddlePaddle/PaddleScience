"""
simple helper functions for reading and
saving CSV files
"""

import csv
import numpy as np


def csv_to_dict(filename, mapping=None, delimiter=","):
    """
    reads a csv file to a dictionary of columns

    Parameters
    ----------
    filename : str
      The file name to load from
    mapping : None, dict
      If None load entire csv file and store
      every column as a key in the dict. If
      `mapping` is not none use this to map
      keys from CSV to keys in dict.
    delimiter: str
      The string used for separating values.

    Returns
    -------
    data : dict of numpy arrays
      numpy arrays have shape [N, 1].
    """

    # Load csv file
    values = np.loadtxt(filename, skiprows=1, delimiter=delimiter, unpack=False)

    # get column keys
    csvfile = open(filename)
    reader = csv.reader(csvfile, delimiter=delimiter)
    first_line = next(iter(reader))

    # set dictionary
    csv_dict = {}
    for i, name in enumerate(first_line):
        if mapping is not None:
            if name.strip() in mapping.keys():
                csv_dict[mapping[name.strip()]] = values[:, i : i + 1]
        else:
            csv_dict[name.strip()] = values[:, i : i + 1]
    return csv_dict


def dict_to_csv(dictonary, filename):
    """
    saves a dict of numpy arrays to csv file

    Parameters
    ----------
    dictionary : dict
      dictionary of numpy arrays. The numpy
      arrays have a shape of [N, 1].
    filename : str
      The file name to save too
    """

    # add csv to filename
    if filename[-4:] != ".csv":
        filename += ".csv"

    # save np arrays
    csvfile = open(filename, "w+")
    csvfile.write(",".join(['"' + str(x) + '"' for x in list(dictonary.keys())]) + "\n")
    for i in range(next(iter(dictonary.values())).shape[0]):
        csvfile.write(",".join([str(x[i, 0]) for x in dictonary.values()]) + "\n")
    csvfile.close()
