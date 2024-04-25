import numpy as np
import pandas as pd
from collections import defaultdict
import os
from typing import (NamedTuple)

from random import shuffle

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
#from category_encoders import TargetEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import train_test_split
from sklearn.base import clone

class Data(NamedTuple):
    attributes: list
    sample_class: list
def read_amphibians_data_():
    directory_path = "data_sets/amphibians/"
    file_name = "dataset2.csv"          #uważać jaki plik wczytujemy!!!!
    filepath = os.path.join(directory_path, file_name)
    data = []
    with open(filepath, 'r') as file:
        next(file)  # Pomija pierwszą linię
        next(file)  # Pomija drugą linię
        for line in file:
            line = line.split(';')
            line = [int(i) for i in line]
            attributes = line[4:11] + line[13:16]
            sample_class = line[16:]
            data.append(Data(attributes, sample_class))
    return data
def split_amphibians_data(data):
    shuffle(data)
    train_size = 4 * len(data) // 5
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
def read_anuran_data_():
    pass
def get_X_y(data):
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i].attributes)
        y.append(data[i].sample_class)
    X_array = np.array(X)
    y_array = np.array(y)
    assert len(X) == len(y), f"Liczba próbek w X ({len(X)}) nie jest taka sama jak liczba etykiet w y ({len(y)})"
    return X_array, y_array
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

    X_array = np.array(X)
    y_array = np.array(y)

    Xt_array = np.array(Xt)
    yt_array = np.array(yt)

    X_encoded = OneHotEncoder().fit_transform(X_array).toarray()
    Xt_encoded = OneHotEncoder().fit_transform(Xt_array).toarray()

    y_encoded = y_array
    yt_encoded = yt_array

    return X_encoded, y_encoded, Xt_encoded, yt_encoded

class ClassifiersChain:
    def __init__(self, classifier, order=None):
        self.cls = classifier   #ustalenie bazowego klasyfikatora
        self.order = order      #ustalenie kolejności etykiety do klasyfikacji

    #metoda trenująca łańcuch klasyfikatorów
    def fit(self, X, y):
        #ustalenie kolejności etykiety do klasyfikacji
        if self.order is None:
            self.order = list(range(y.shape[1]))  # defaultowa kolejnosc

        #lista klasyfikatorów do klasyfikacji poszczególnych etykiet łańcucha
        self.models = [clone(self.cls) for _ in range(y.shape[1])]

        #stworzenie kopii X, na której będą przeprowadzane operacje
        X_joined = X.copy()

        #stworzenie nowego dataframe z X i y w ustalonej kolejności
        for val in self.order:
            #X_joined = pd.concat([X_joined, y[val]], axis=1)
            X_joined = np.hstack((X_joined, y[:, val].reshape(-1, 1)))

        #trenowanie każdego klasyfikatora w łańcuchu
        for chain_number, model in enumerate(self.models):
            X_ = X_joined[:, :(X.shape[1] + chain_number)] #wywbarnie zestawu atrybutow
            y_ = y[:, self.order[chain_number]] #wybranie etykiety do klasyfikacji
            model.fit(X_, y_)   #trenowanie klasyfikatora na wybranych danych

    #metoda dokonująca predykcji nieznanego zbioru danych
    def predict(self, X):
        #sprawdzenie czy lista modeli została zainicjalizowana
        if self.models is None:
            raise ValueError("Error. You cannot predict class without training")

        lines = X.shape[0]
        rows = len(self.order)
        pred_chain = np.zeros((lines, rows)) #dataframe dla predykcji
        #pred_probs = np.zeros((lines, rows)) #dataframe dla prawdopodobieństw predykcji klas etykiet
        X_copy = X.copy()
        X_joined = X.copy()

        #X_joined.reset_index(drop=True, inplace=True)
        #X_copy.reset_index(drop=True, inplace=True)

        for chain_number, model in enumerate(self.models):
            if chain_number > 0:  #'''(prev_preds.size)'''
                prev_preds = pred_chain[:, chain_number-1] #wybór poprzednich predykcji (pytanie czy tutaj powinienem brać wszystkie czy tylko tą poprzednią?? tutaj chyba wystarczy usunac ten dwukropek)
                X_joined = np.hstack((X_joined, prev_preds.reshape(-1, 1)))
            #X_joined = pd.concat([X_copy, pd.DataFrame(prev_preds)], axis=1) #dodanie poprzednich predykcji do X
            pred = model.predict(X_joined) #predykcja klasy etykiety
            pred_chain[:, chain_number] = pred #zapisanie predykcji
        a = 5
        return pred_chain

class NaiveBayes:
    def __init__(self):
        self.prior = None #prawdopodobienstwo a'priori
        self.likelihood = None #prawdopodobienstwo warunkowe

    def fit(self, X, y, smoothing=1):
        self.prior = defaultdict(int)
        self.likelihood = {}

        for label in y:
            self.prior[label] += 1 #liczba wystapien etykiet klas

        for label in np.unique(y): #prawdopodobieństwa warunkowe dla poszczególnych cech
            indices = np.where(y == label)[0]
            self.likelihood[label] = (X[indices].sum(axis=0) + smoothing) / (len(indices) + 2 * smoothing)

    def predict(self, X):
        if self.prior is None or self.likelihood is None:
            raise ValueError("Error. You cannot predict class without training")

        posteriors = []
        for x in X:
            posterior = self.prior.copy()
            for label, likelihood_label in self.likelihood.items():
                for i, bool_value in enumerate(x):
                    posterior[label] *= likelihood_label[i] if bool_value else (
                            1 - likelihood_label[i])
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())

        return np.array([max(prediction, key=prediction.get)
                         for prediction in posteriors])
def compare_Bayes(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def main():
    data = read_amphibians_data_()

    #X_train, y_train, X_test, y_test = encode_data(train_data, test_data)


    gnb = GaussianNB()
    cnb = CategoricalNB()
    mnb = MultinomialNB()
    a=0
    for _ in range(10):
        train_data, test_data = split_amphibians_data(data)
        X_train, y_train = get_X_y(train_data)
        X_test, y_test = get_X_y(test_data)
        y_pred = mnb.fit(X_train, y_train[:, 4]).predict(X_test)
        #print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test[:, 0] != y_pred).sum()))
        #print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test[:,4], y_pred)))
        a = a + accuracy_score(y_test[:, 4], y_pred)
    print("Hello, World!")
    print(a/10)

    a = y_pred
    b = y_test[:, 1]

    chain_classifier = ClassifiersChain(mnb)
    chain_classifier.fit(X_train, y_train)
    y_pred = chain_classifier.predict(X_test)
    print(y_pred)
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()