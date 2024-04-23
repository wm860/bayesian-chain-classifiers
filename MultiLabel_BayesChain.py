import numpy as np
from collections import defaultdict
import os
from typing import (NamedTuple)

from random import shuffle

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


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

class BayesianChainClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X_train, y_train):
        for i, clf in enumerate(self.classifiers):
            if i == 0:
                clf.fit(X_train, y_train)
            else:
                X_train, y_train = self._generate_input_data(X_train, y_train, i-1)
                clf.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = None
        for i, clf in enumerate(self.classifiers):
            if i == 0:
                y_pred = clf.predict(X_test)
            else:
                X_test = self._generate_input_data(X_test, y_pred, i-1)
                y_pred = clf.predict(X_test)
        return y_pred

    def _generate_input_data(self, X, y_pred, index):
        # Przygotuj dane wejściowe dla następnego klasyfikatora
        # Tutaj można dodać logikę modyfikacji danych na podstawie poprzednich predykcji
        return X

# Przygotuj dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiuj klasyfikatory Bayesa w łańcuchu
classifiers = [
    MultinomialNB(),
    MultinomialNB()
]

# Inicjalizuj i trenuj łańcuch klasyfikatorów Bayesa
chain_classifier = BayesianChainClassifier(classifiers)
chain_classifier.fit(X_train, y_train)

# Dokonaj predykcji
y_pred = chain_classifier.predict(X_test)


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
    train_data, test_data = split_amphibians_data(data)

    print("Hello, World!")

if __name__ == "__main__":
    main()