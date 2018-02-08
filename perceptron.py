# Import libraries and classes
from memory_profiler import profile
import utility as utl
import numpy as np


# Simple perceptron train function
@profile
def perceptron_train(data, labels, max_iter):
    w = np.zeros(len(data[0]))
    b = 0
    for iter in range(max_iter):
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            if y * (np.dot(x, w) + b) <= 0:
                delta = np.multiply(y, x)
                w = np.add(w, delta)
                b += y
    return w, b


# Simple perceptron prediction function
def perceptron_predict(w, b, example):
    return np.sign(np.dot(example, w) + b)


# Voted perceptron train function
@profile
def voted_perceptron_train(data, labels, max_iter):
    w = np.zeros(len(data[0]))
    b = 0
    time_alive = 1
    perceptrons = []
    for iter in range(max_iter):
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            if y * (np.dot(x, w) + b) <= 0:
                perceptrons.append(utl.join(w, b, time_alive))
                time_alive = 1
                delta = np.multiply(y, x)
                w = np.add(w, delta)
                b += y
            else:
                time_alive += 1
    perceptrons.append(utl.join(w, b, time_alive))
    return perceptrons


# Voted perceptron prediction function
def voted_perceptron_predict(perceptrons, example):
    sum = 0
    for perceptron in perceptrons:
        w, b, weight = utl.extract_data(perceptron)
        sum += weight * np.sign(np.dot(w, example))
    return np.sign(sum)


# The classifier is trained and than tested. Additional information is displayed.
# Tp: TruePositive, Tn: TrueNegative, Fp: FalsePositive, Fn: FalseNegative
def test_default(categories, max_iter=10, seed=42):
    print "Standard Perceptron Test."
    print "Categories: ", categories[0], " and ", categories[1]
    (train_data, test, test_data, labels) = utl.get_input_from_text(categories, seed=seed)
    (w, b) = perceptron_train(train_data, labels, max_iter)
    tp = tn = fp = fn = 0
    for i in range(len(test_data)):
        result = perceptron_predict(w, b, test_data[i])
        if result == -1 and test.target[i] == 0:
            tn += 1
        elif result == 1 and test.target[i] == 1:
            tp += 1
        elif result == -1 and test.target[i] == 1:
            fn += 1
        else:
            fp += 1
    print "Confusion matrix: "
    print utl.print_confusion_matrix(categories, tp, tn, fp, fn)
    print "Accuracy: ", float((tp + tn)) / (tp + tn + fn + fp) * 100, "%"


# The classifier is trained and than tested. Additional information is displayed.
def test_voted(categories, max_iter=10, seed=42):
    print "Voted Perceptron Test."
    print "Categories: ", categories[0], " and ", categories[1]
    (train_data, test, test_data, labels) = utl.get_input_from_text(categories, seed=seed)
    perceptrons = voted_perceptron_train(train_data, labels, max_iter)
    tp = tn = fp = fn = 0
    for i in range(len(test_data)):
        result = voted_perceptron_predict(perceptrons, test_data[i])
        if result == -1 and test.target[i] == 0:
            tn += 1
        elif result == 1 and test.target[i] == 1:
            tp += 1
        elif result == -1 and test.target[i] == 1:
            fn += 1
        else:
            fp += 1
    print "Confusion matrix: "
    print utl.print_confusion_matrix(categories, tp, tn, fp, fn)
    print "Accuracy: ", float((tp + tn)) / (tp + tn + fn + fp) * 100, "%"
    print "Number of stored classifiers: ", len(perceptrons)
