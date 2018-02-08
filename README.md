# Classificazione Testuale mediante Perceptron

Il progetto ha come scopo quello di mettere in luce proprietà e criticità dell'algoritmo Perceptron, qui utilizzato per classificare su base binaria le classi del dataset 20 Newsgroup. Per migliorarne l'accuratezza è stata poi implementata una versione votata che tiene conto del peso di ciascun piano.

## Prerequisiti

Per poter eseguire correttamente il codice sono necessari:

* [Scikit-Learn](http://scikit-learn.org/stable/index.html#) da cui è possibile ottenere il dataset 20 Newsgroup e diverse funzionalità per trattare il testo trasformandolo in input numerico.
* [Numpy](http://www.numpy.org/) necessario ad eseguire le operazioni sui vettori.
* [Memory Profiler](https://pypi.python.org/pypi/memory_profiler) utile per tenere traccia dell'occupazione di memoria.
* [Pretty Table](https://pypi.python.org/pypi/PrettyTable) per formattare al meglio la matrice di confusione.


## Esecuzione

L'interfaccia da cui è possibile compiere esperimenti è costituita dalla classe `test.py`. Sono stati riportate tre coppie di categorie di esempio, ma in generale possono essere scelte a piacere da tutte quelle disponibili:

* comp.os.ms-windows.misc
* comp.sys.ibm.pc.hardware
* comp.sys.mac.hardware
* comp.windows.x
* rec.autos
* rec.motorcycles
* rec.sport.baseball
* rec.sport.hockey
* sci.crypt
* sci.electronics
* sci.med
* sci.space
* misc.forsale
* talk.politics.misc
* talk.politics.guns
* talk.politics.mideast
* talk.religion.misc
* alt.atheism
* soc.religion.christian

Nelle principali funzioni è possibile modificare il parametro `max_iter` per variare il numero di volte che si ciclerà sui dati di training e `seed` per cambiarne l'ordinamento.
```python
perceptron.test_default(categories, max_iter=10, seed=8)
perceptron.test_voted(categories, max_iter=10, seed=8)
```

Sebbene sia sconsigliato, all'interno della classe `util.py` è possible includere anche ulteriori elementi del testo originale come headers, footers e quotes rimuovendo l'ultimo attributo.

```python
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                               random_state=seed, remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', categories=categories,
                              remove=('headers', 'footers', 'quotes'))
```

Se si volesse ottenere un dettaglio grafico dell'utilizzo della memoria è necessario eseguire

```
mprof run test.py
mprof plot
```

## Riferimenti

Nella realizzazione del progetto sono stati consultati:

* [Large Margin Classification Using the Perceptron Algorithm](http://cseweb.ucsd.edu/~yfreund/papers/LargeMarginsUsingPerceptron.pdf) di Y. Freund e R.E. Schapire
* [Studio sul dataset 20 Newsgroup e sue caratteristiche](http://cn-static.udacity.com/mlnd/Capstone_Poject_Sample01.pdf)
* [Documentazione Scikit-Learn sulla classificazione testuale](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [A Course in Machine Learning](http://ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf) di Hal Daumé

