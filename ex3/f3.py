# in here this script will generate random labels for each data set and then compared it with actual value.
# you'll see it's low, because it just put label randomly without considering any features 

import random

from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class ScrappyKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train	# create local variable for the class
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = random.choice(self.y_train)	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
			predictions.append(label)
		return predictions

iris = datasets.load_iris()

x = iris.data 	# data related to features
y = iris.target	# label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# my_classifier = KNeighborsClassifier()

my_classifier = ScrappyKNN()
my_classifier.fit(x_train, y_train)				# my_classifier object is going to be acutomatically pass as self parameter.
												# so "self" is temporary place hoslder for object itself.
predictions = my_classifier.predict(x_test)
  
print (accuracy_score(y_test,predictions))
