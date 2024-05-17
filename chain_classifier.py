import numpy as np
import pandas as pd
from sklearn.base import clone

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

'''
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
            if chain_number > 0:  # (prev_preds.size)
                prev_preds = pred_chain[
                    :, chain_number - 1
                ]  # wybór poprzednich predykcji (pytanie czy tutaj powinienem brać wszystkie czy tylko tą poprzednią?? tutaj chyba wystarczy usunac ten dwukropek)
                X_joined = np.hstack((X_joined, prev_preds.reshape(-1, 1)))
            # X_joined = pd.concat([X_copy, pd.DataFrame(prev_preds)], axis=1) #dodanie poprzednich predykcji do X
            pred = model.predict(X_joined)  # predykcja klasy etykiety
            pred_chain[:, chain_number] = pred  # zapisanie predykcji
        a = 5
        return pred_chain
'''