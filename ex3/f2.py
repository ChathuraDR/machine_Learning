from sklearn import datasets

#### partition the dataset into train and set ####
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

### classifire ###
#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

### compare the predicted labels with true lables
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()

x = iris.data 	# data related to features
y = iris.target	# lables

#
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)		# test size will be exactly half of the data set

#################################################
#my_classifier = tree.DecisionTreeClassifier()

my_classifier = KNeighborsClassifier()	# another classifier like DecisionTree

my_classifier.fit(x_train, y_train)
predictions = my_classifier.predict(x_test)

#################################################
print (accuracy_score(y_test,predictions))
