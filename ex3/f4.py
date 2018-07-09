# Now let's try to improve the result like we get in using actual classifier in f2 example.
# This is based on k-NN (k-Nearst Neighbors)
# Mathematical calcualations = > ecuclidean distance to measure nearest neighbours

from scipy.spatial import distance 
########################################################
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#return the distanse between two points, using euclidean algorithm ( a bit like pythagorean therom)
def euc(a,b):
	# gives by fowlloing formula
	# d(a,b) = ((x2-x1)^2 + (y2-y1)^2)^1/2
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train	# create local variable for the class
		self.y_train = y_train
	
	# defines the closest training point to the test point 
	def closest(self, row):
		best_dist = euc(row, self.x_train[0])	# keep track of the shortest distance
		best_index = 0	# keep track of the index of the training point
		for i in range(1,len(self.x_train)):
			dist = euc(row, self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

	def predict(self, x_test):
		predictions = []
		for row in x_test:
			label = self.closest(row)	# find the closest point to the test point
			predictions.append(label)
		return predictions

iris = datasets.load_iris()

x = iris.data 	# data related to features
y = iris.target	# label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

my_classifier = ScrappyKNN()
my_classifier.fit(x_train, y_train)				

predictions = my_classifier.predict(x_test)
  
print (accuracy_score(y_test,predictions))