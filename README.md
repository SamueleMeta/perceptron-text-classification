# Perceptron Text Classification

<p align="center">
    <img src="https://i.imgur.com/lgi4uSY.jpg" width="250" alt="Università degli Studi Firenze"/>
</p>

## Overview

Despite the fact that the standard Perceptron provides a good result in its simplicity, it presents some criticalities. Suppose, for example, that we want to classify the XOR function. It is immediately evident that it is impossible to draw a plane that divides the positive examples from the negative ones without committing any error. It follows that the algorithm, not being data linearly separable, will continue to generate each time a different plan and the final one will be randomly determined by the moment in which the stop occurs after a certain number of iterations. 

<p align="center">
    <img src="https://i.imgur.com/eO6M36Z.png" width="250" alt="xor function" />
</p>
       
Suppose now to train the Perceptron and obtain, after a few iterations, a satisfactory classifier that correctly predicts the next 5000 submitted data points. If the last datum is classified incorrectly, the plan must be updated despite its previous accuracy. To limit these situations, whenever a plan has to change, the number _c_ of correct consecutive classifications will be saved. In this way, during testing, it will be possible to possible to determine the sign of an example by weighing the contribution of each plan, according to the formula:
    
<p align="center">
    <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^k&space;c_i&space;\cdot&space;sign(\vec{w}&space;\cdot&space;\vec{x})" target="_blank"><img      src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^k&space;c_i&space;\cdot&space;sign(\vec{w}&space;\cdot&space;\vec{x})" title="\sum_{i=1}^k c_i \cdot sign(\vec{w} \cdot \vec{x})" /></a>
</p>

The experiments revealed, as expected, dependence on the order in which the data were shown as input. This implies that, for the same problem, different seeds can generate very different performance for the standard version, while the voted one remains stable. More details, schematized as the table below, can be found in the final report.

<p align="center">
    <img src="https://i.imgur.com/zvRXXsQ.png" width="500" alt="result table" />
</p>


## Prerequisites

* [Scikit-Learn](http://scikit-learn.org/stable/index.html#) to obtain the 20 Newsgroup dataset and various functionalities to transform the text into a numeric input.
* [Numpy](http://www.numpy.org/) to perform vectorized operations.
* [Memory Profiler](https://pypi.python.org/pypi/memory_profiler) useful to keep trace of memory occupation.
* [Pretty Table](https://pypi.python.org/pypi/PrettyTable) for a nice confusion matrix formatting.


## Run

Experiments can be launched from the `test.py` file, containing three category couples as an example. In general, it is possible to choose them from the following list:

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

In the two main functions it is possible to change the `max_iter` and `seed` parameters, in order to affect the number of cycles on the training data and to obtain different scenarios according to the shuffling.

```python
perceptron.test_default(categories, max_iter=10, seed=8)
perceptron.test_voted(categories, max_iter=10, seed=8)
```

Although it is not recommended, within the ```util.py``` class it is possible to include additional elements of the original text such as headers, footers and quotes by removing the last attribute.

```python
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                               random_state=seed, remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', categories=categories,
                              remove=('headers', 'footers', 'quotes'))
```

If you want to get a graphical detail of the memory usage you need to run

```
mprof run test.py
mprof plot
```

## References

* [Large Margin Classification Using the Perceptron Algorithm](http://cseweb.ucsd.edu/~yfreund/papers/LargeMarginsUsingPerceptron.pdf) by Y. Freund and R.E. Schapire
* [Working with text data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) from Scikit-Learn documentation
* [A Course in Machine Learning](http://ciml.info/dl/v0_99/ciml-v0_99-ch04.pdf) by Hal Daumé

