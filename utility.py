from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from prettytable import PrettyTable


# Utility function to select the categories and fetch the data.
# Headers, footers and quotes are remove from text to improve generalization.
# CountVectorizer creates a bag of words from the text data
# Term-frequency times inverse document-frequency transformation is applied.
def get_input_from_text(categories, seed=42):
    train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True,
                               random_state=seed, remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', categories=categories,
                              remove=('headers', 'footers', 'quotes'))
    cv = CountVectorizer(stop_words='english')
    train_vectorized = cv.fit_transform(train.data)
    test_vectorized = cv.transform(test.data)
    tfidf = TfidfTransformer()
    train_transformed = tfidf.fit_transform(train_vectorized)
    test_transformed = tfidf.transform(test_vectorized)
    train_data = train_transformed.toarray()
    test_data = test_transformed.toarray()
    labels = train.target[:]
    labels[labels == 0] = -1
    return train_data, test, test_data, labels


# Utility function to join relevant information all together
def join(w, b, time_alive):
    perceptron = []
    perceptron.append(w)
    perceptron.append(b)
    perceptron.append(time_alive)
    return perceptron


# Utility function to extract data from perceptron's list
def extract_data(perceptron):
    w = perceptron[0]
    b = perceptron[1]
    weight = perceptron[2]
    return w, b, weight


# Prints the confusion matrix. Tp: TruePositive ecc.
def print_confusion_matrix(categories, tp, tn, fp, fn):
    t = PrettyTable(['\\', categories[0], categories[1]])
    t.add_row([categories[0], tn, fp])
    t.add_row([categories[1], fn, tp])
    return t

