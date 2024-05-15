import numpy as np
import pandas as pd
import os
from typing import NamedTuple

from random import shuffle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from chow_liu_py import chow_liu

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


class ClassifiersChain_array:
    def __init__(self, classifier, order=None):
        self.cls = classifier  # ustalenie bazowego klasyfikatora
        self.order = order  # ustalenie kolejności etykiety do klasyfikacji

    # metoda trenująca łańcuch klasyfikatorów
    def fit(self, X, y):
        Xa = np.array(X)
        ya = np.array(y)
        # ustalenie kolejności etykiety do klasyfikacji
        if self.order is None:
            self.order = list(range(ya.shape[1]))  # defaultowa kolejnosc

        # lista klasyfikatorów do klasyfikacji poszczególnych etykiet łańcucha
        self.models = [clone(self.cls) for _ in range(ya.shape[1])]

        # stworzenie kopii X, na której będą przeprowadzane operacje
        X_joined = Xa.copy()

        # stworzenie nowego dataframe z X i y w ustalonej kolejności
        for val in self.order:
            # X_joined = pd.concat([X_joined, y[val]], axis=1)
            X_joined = np.hstack((X_joined, ya[:, val].reshape(-1, 1)))

        # trenowanie każdego klasyfikatora w łańcuchu
        for chain_number, model in enumerate(self.models):
            X_ = X_joined[:, : (Xa.shape[1] + chain_number)]  # wywbarnie zestawu atrybutow
            y_ = ya[:, self.order[chain_number]]  # wybranie etykiety do klasyfikacji
            model.fit(X_, y_)  # trenowanie klasyfikatora na wybranych danych

    # metoda dokonująca predykcji nieznanego zbioru danych
    def predict(self, X):
        # sprawdzenie czy lista modeli została zainicjalizowana
        if self.models is None:
            raise ValueError("Error. You cannot predict class without training")

        lines = X.shape[0]
        rows = len(self.order)
        pred_chain = np.zeros((lines, rows))  # dataframe dla predykcji
        # pred_probs = np.zeros((lines, rows)) #dataframe dla prawdopodobieństw predykcji klas etykiet
        X_copy = X.copy()
        X_joined = X.copy()

        # X_joined.reset_index(drop=True, inplace=True)
        # X_copy.reset_index(drop=True, inplace=True)

        for chain_number, model in enumerate(self.models):
            if chain_number > 0:  # '''(prev_preds.size)'''
                prev_preds = pred_chain[
                    :, chain_number - 1
                ]  # wybór poprzednich predykcji (pytanie czy tutaj powinienem brać wszystkie czy tylko tą poprzednią?? tutaj chyba wystarczy usunac ten dwukropek)
                X_joined = np.hstack((X_joined, prev_preds.reshape(-1, 1)))
            # X_joined = pd.concat([X_copy, pd.DataFrame(prev_preds)], axis=1) #dodanie poprzednich predykcji do X
            pred = model.predict(X_joined)  # predykcja klasy etykiety
            pred_chain[:, chain_number] = pred  # zapisanie predykcji
        a = 5
        return pred_chain


class ClassifiersChain:
    def __init__(self, classifier, order=None):
        self.cls = classifier  # ustalenie bazowego klasyfikatora
        self.order = order  # ustalenie kolejności etykiety do klasyfikacji

    # metoda trenująca łańcuch klasyfikatorów
    def fit(self, X, y):
        # Xa = np.array(X)
        # ya = np.array(y)
        # ustalenie kolejności etykiety do klasyfikacji
        if self.order is None:
            self.order = list(range(y.shape[1]))  # defaultowa kolejnosc

        # lista klasyfikatorów do klasyfikacji poszczególnych etykiet łańcucha
        self.models = [clone(self.cls) for _ in self.order]

        self.col_names = y.columns[self.order].tolist()
        X.reset_index(drop=True, inplace=True)
        y_new = y[self.col_names]
        X_joined = pd.concat([X, y_new], axis=1)

        # trenowanie każdego klasyfikatora w łańcuchu
        for chain_number, model in enumerate(self.models):
            X_ = X_joined.iloc[:, : (X.shape[1] + chain_number)]  # wybranie zestawu atrybutow
            y_ = y_new[self.col_names[chain_number]]  # wybranie etykiety do klasyfikacji
            model.fit(X_, y_)  # trenowanie klasyfikatora na wybranych danych

    # metoda dokonująca predykcji nieznanego zbioru danych
    def predict(self, X):
        # sprawdzenie czy lista modeli została zainicjalizowana
        if self.models is None:
            raise ValueError("Error. You cannot predict class without training")

        X_joined = X.copy()
        data_types = {col: np.int32 for col in self.col_names}
        pred_chain = pd.DataFrame(columns=data_types.keys()).astype(data_types)

        for chain_number, model in enumerate(self.models):
            if chain_number > 0:
                prev_preds = pred_chain.iloc[
                    :, chain_number - 1
                ]  # wybór poprzednich predykcji (pytanie czy tutaj powinienem brać wszystkie czy tylko tą poprzednią?? tutaj chyba wystarczy usunac ten dwukropek)
                X_joined = pd.concat([X_joined, prev_preds], axis=1)
            pred = model.predict(X_joined)  # predykcja klasy etykiety
            pred_chain.iloc[:, chain_number] = pred  # zapisanie predykcji
        return pred_chain.astype(np.int32)


def main():
    # data = read_amphibians_data()
    X, y = read_anuran_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    gnb = GaussianNB()
    cnb = RandomForestClassifier()  # CategoricalNB()
    mnb = MultinomialNB()

    f1s = []
    acs = []

    ''' kod odpowiedzialny za trenowanie, predykcje i ocene pojedynczego klasyfikatora
    class_index = 0
    class_name = "Family"
    for _ in range(1):
        y_pred = gnb.fit(X_train, y_train[class_name]).predict(X_test)
        # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test[:, 0] != y_pred).sum()))
        print("Model accuracy score: {0:0.4f}".format(accuracy_score(y_test[class_name], y_pred)))
        # f1s.append(f1_score(y_test[:, class_index], y_pred))  # accuracy_score(y_test[:, 0], y_pred)
        acs.append(accuracy_score(y_test[class_name], y_pred))

    print("classifier accuracy score: ", np.mean(acs))
    # print("F1 score: ", np.mean(f1s))
    '''
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # kodowanie etykiet
    y_train_encoded, y_test_encoded = encode_etiquette(y_train, y_test)
    y_test_encoded.reset_index(drop=True, inplace=True)

    ''' sprawdzenie dizalania łańcucha klasyfikatorów
    for _ in range(1):
        chain_classifier = ClassifiersChain(gnb, order=[2, 1, 0])
        chain_classifier.fit(X_train, y_train_encoded)
        y_pred = chain_classifier.predict(X_test)
        print(class_name)
        print("chain score ", accuracy_score(y_test_encoded[class_name], y_pred[class_name]))
        print("F1 chain: ", f1_score(y_test_encoded[class_name], y_pred[class_name], average="macro"))
    '''

    res = []
    res = chow_liu(y_train_encoded, root="Genus")
    print(5)


if __name__ == "__main__":
    main()
