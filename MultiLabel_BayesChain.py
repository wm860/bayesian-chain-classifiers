import numpy as np
from collections import defaultdict
import os
from typing import (NamedTuple)

from random import shuffle
from sklearn.preprocessing import OneHotEncoder


class Data(NamedTuple):
    attributes: list
    sample_class: list

def read_amphibians_data_():
    directory_path = "data_sets/amphibians/"
    file_name = "dataset.csv"
    filepath = os.path.join(directory_path, file_name)
    data = []
    with open(filepath, 'r') as file:
        next(file)  # Pomija pierwszą linię
        next(file)  # Pomija drugą linię
        for line in file:
            line = line.split(';')
            attributes = line[2:16]
            sample_class = line[16:]
            data.append(Data(attributes,sample_class))
    return data

def split_amphibians_data(data):
    shuffle(data)
    train_size = 4 * len(data) // 5
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
def read_anuran_data_():
    pass




def main():
    data = read_amphibians_data_()
    train_data, test_data = split_amphibians_data(data)

    print("Hello, World!")

if __name__ == "__main__":
    main()