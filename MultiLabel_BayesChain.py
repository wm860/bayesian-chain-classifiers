from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from chow_liu_py import chow_liu
from preprocessing import *
from chain_classifier import ClassifiersChain


def main():
    # data = read_amphibians_data()
    X, y = read_anuran_data()
    # print(X.shape[0])
    # print(X.shape[1])
    # print(y.shape[0])
    # print(y.shape[1])
    # for col in y.columns:
    #     print(y[col].value_counts())
    #     print(" ")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    gnb = GaussianNB()
    cnb = RandomForestClassifier()  # CategoricalNB()
    mnb = MultinomialNB()

    f1s = []
    acs = []


    class_index = 0
    class_name = "Family"
    #kod odpowiedzialny za trenowanie, predykcje i ocene pojedynczego klasyfikatora
    for _ in range(1):
        y_pred = gnb.fit(X_train, y_train[class_name]).predict(X_test)
        # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test[:, 0] != y_pred).sum()))
        print("Model accuracy score: {0:0.4f}".format(accuracy_score(y_test[class_name], y_pred)))
        # f1s.append(f1_score(y_test[:, class_index], y_pred))  # accuracy_score(y_test[:, 0], y_pred)
        acs.append(accuracy_score(y_test[class_name], y_pred))

    print("classifier accuracy score: ", np.mean(acs))
    # print("F1 score: ", np.mean(f1s))

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    # kodowanie etykiet
    y_train_encoded, y_test_encoded = encode_etiquette(y_train, y_test)
    y_test_encoded.reset_index(drop=True, inplace=True)

    # bayesian chain
    y_train_encoded_numerical_columns = y_train_encoded.copy()
    y_train_encoded_numerical_columns.columns = range(y_train_encoded_numerical_columns.shape[1])
    res = []
    res = chow_liu(y_train_encoded_numerical_columns, root=1)

    #łańcuch klasyfikatorów
    chains = pd.DataFrame(columns=y_test_encoded.columns)
    for i in range(len(res)):
        chain_classifier = ClassifiersChain(gnb, order=list(res[i]))
        chain_classifier.fit(X_train, y_train_encoded)
        y_pred = chain_classifier.predict(X_test)
        if i == 0:
            chains = pd.concat([chains, y_pred])
        else:
            chains = chains.fillna(y_pred)
        chains = chains.astype(int, errors='ignore')

    for name in y_test.columns:
        print(name)
        print("chain score ", accuracy_score(chains[name], y_test_encoded[name]))
        print("F1 chain: ", f1_score(chains[name], y_test_encoded[name], average="macro"))


if __name__ == "__main__":
    main()
