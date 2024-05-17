import numpy as np
import pandas as pd
import os
from typing import NamedTuple

from random import shuffle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class Data(NamedTuple):
    attributes: list
    sample_class: list


def read_amphibians_data():
    directory_path = "data_sets/amphibians/"
    file_name = "dataset2.csv"  # uważać jaki plik wczytujemy!!!!
    filepath = os.path.join(directory_path, file_name)
    data = []
    with open(filepath, "r") as file:
        next(file)  # Pomija pierwszą linię
        next(file)  # Pomija drugą linię
        for line in file:
            line = line.split(";")
            line = [int(i) for i in line]
            attributes = line[4:11] + line[13:16]
            sample_class = line[16:]
            data.append(Data(attributes, sample_class))
    return data


def read_anuran_data():
    df = pd.read_csv("data_sets/anuran/Frogs_MFCCs.csv", index_col=False)
    df = df.drop(columns=["RecordID"])
    n_classes = 3
    return df.iloc[:, :-n_classes], df.iloc[:, -n_classes:]


def split_data(data):
    shuffle(data)
    train_size = 4 * len(data) // 5
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def get_X_y(data):
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i].attributes)
        y.append(data[i].sample_class)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # X_array = np.array(X)
    # y_array = np.array(y)
    # assert len(X) == len(y), f"Liczba próbek w X ({len(X)}) nie jest taka sama jak liczba etykiet w y ({len(y)})"
    return X, y


def encode_data(train_data, test_data):
    X = []
    Xt = []
    y = []
    yt = []
    for i in range(len(train_data)):
        X.append(train_data[i].attributes)
        y.append(train_data[i].sample_class)
    for i in range(len(test_data)):
        Xt.append(test_data[i].attributes)
        yt.append(test_data[i].sample_class)

    onh = OneHotEncoder(handle_unknown="ignore")

    X_array = np.array(X)
    y_array = np.array(y)

    Xt_array = np.array(Xt)
    yt_array = np.array(yt)

    onh.fit(X_array)
    X_encoded = onh.transform(X_array).toarray()
    Xt_encoded = onh.transform(Xt_array).toarray()

    y_encoded = y_array
    yt_encoded = yt_array

    return X_encoded, y_encoded, Xt_encoded, yt_encoded


def encode_etiquette(y_train, y_test):
    le = LabelEncoder()
    y_train_encoded = pd.DataFrame()
    y_test_encoded = pd.DataFrame()
    for col in y_train.columns:
        y_train_encoded[col] = le.fit_transform(y_train[col])
        y_test_encoded[col] = le.transform(y_test[col])
    return y_train_encoded, y_test_encoded