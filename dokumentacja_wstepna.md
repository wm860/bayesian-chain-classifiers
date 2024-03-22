# **Dokumentacja wstępna ZUM**

*Autorzy: Wojciech Makos, Jakub Kliszko*


## Temat:  
Tworzenie modeli klasyfikacji wieloetykietowej przez zastosowanie dekompozycji na wiele powiązanych zadań klasyfikacji jednoetykietowych zgodnie z metodą bayesowskiego łańcucha klasyfikatorów. Porównanie z algorytmami klasyfikacji wieloetykietowej dostępnymi w środowisku R lub Python. (Bayes classifier chain for multi-label classification)

### 1. Interpretacja tematu projektu

Głównym celem projektu jest implementacja algorytmu służącego do klasysyfikacji przykładów posiadających więcej niż jedną etykietę. Każdy przykład ma przypisany zestaw binarnych znaczników. Jednym ze sposobów stworzenia modelu umożliwiającego predykcje na bazie przykładów wieloetykietowych jest dokonanie tranformacji tej bazy zgodnie z metodą łańcucha klasyfikatorów. Metoda ta polega na sekwencyjnym (czyli w kolejności) tworzeniu modeli klasyfikacji binarnej, w której to dla każdego kolejnego modelu dodaje się do zbioru atrybutów predykcje etykiet poprzednich modeli wykorzystanych przy wcześniej rozważanych etykietach. Innymi słowy dla przykładu składającego się z wektora atrybutów X = (x1, x2, x3) i etykiet Y = (y1, y2), pierwszy model tworzymy dla przykładu składającego się jedynie z wektora atrybutów X = (x1, x2, x3) i etykiety Y = (y1). Model przewiduje nam wartość y1. W kolejnym podejściu tworzymy drugi model, tym razem już dla przykładu składającego się z wektora atrybutów X = (x1, x2, x3, y1) i etykiety Y = (y2). Kolejność predykcji następujących po sobie etykiet jest istotna i ma wpływ na końcowy wynik, zatem w testach modelu postaramy się zbadać wpływ kolejności etykiet na wynik klasyfikacji dla konkretnych przykładów. 

### 2. Opis części implementacyjnej oraz lista algorytmów, bibliotek, klas, funkcji


### 3. Plan badań:
   - #### Cel badań,

Z racji na implementacyjny charakter projektu postanowiliśmy zbadać i zweryfikowac jedynie podstawowe czynniki i parametry modelu mogące mieć wpływ na ostateczne wyniki klasyfikacji. Planuje się wyznaczyć: 
   - dokładność wyników klasyfikacji dla przynajmniej 2 zestawów danych testowych, 
   - czas uczenia modelu,,
   - wpływ kolejności etykiet na wynik klasyfikacji,
   - wpływ .... na wynik klasyfikacji,
   - wpływ .... na wynik klasyfikacji,
   - dodatkowo, gdy wystarczy czasu to chcielibyśmy wyznaczyć współczynniki takie, jak: precyzja, czułość, F1 score.
   
   - #### Charakterystyka zbioru danych:

Zdecydowano się na wybór danych testowych z udostępnionego zbioru: https://mulan.sourceforge.net/datasets-mlc.html. 

  - #### Procedura ocenu modeli 
Przede wszystkim planujemy wyznaczyć dokładność predykcji modelu. W tym celu wybrany zbiór danych zostanie podzielony na zbiór treningowy i testowy w proporcji 4:1. Następnie model zostanie wytrenowany na zbiorze treningowym, a wyniki klasyfikacji zostaną porównane z etykietami zbioru testowego. Tą samą operację zamierzamy przeprowadzić dla 2 ziorów danych a dokładność wyników porównać ze sobą oraz z metodami dostępnymi z biblioteki scikit-learn. 

