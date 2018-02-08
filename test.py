import perceptron


# categories = ['rec.autos', 'rec.motorcycles']
# categories = ['sci.crypt', 'sci.electronics']
categories = ['comp.graphics', 'comp.sys.mac.hardware']


perceptron.test_default(categories, max_iter=10, seed=8)
perceptron.test_voted(categories, max_iter=10, seed=8)

