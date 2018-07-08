import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
###VIsualisation###
import graphviz 


iris = load_iris()
test_idx = [0, 50, 100] #testing data set from differnet categories


# numpy.delete(Input array, Indicate which sub-arrays to remove, axis=None)[source]
#
# axis = None will delete only one value from particular array, 
# axit = 0 will delete ther whole raw and axis = 1 will delete the whole column
#
# raw = 0 , column = 1
#

#training data
#
train_target = np.delete(iris.target, test_idx)	# lables
train_data = np.delete(iris.data, test_idx, 0)	# features


#testing data
#
test_target = iris.target[test_idx]	# lables
test_data = iris.data[test_idx]		# features

# time to train the classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print ("Test data actual lables : %s" % (test_target))
print ("Tree predict : %s" % (clf.predict(test_data)))	# What the tree predict

# print (len(train_target))
# print ("##############")
# print (len(train_data))
# print (train_data)

#printing meta data
# print(iris.feature_names)
# print(iris.target_names)

#printing entries,data
#print(iris.data[0])		# data without lables
#print(iris.target[0])



# for i in range(len(iris.target)):
# 	print("Example %d: label %s, features %s" %(i, iris.target[i],iris.data[i]))
# 	


# visualisation
# 
# in order to work properly you might want folloing libararies installed into your linux system
# libcgraph6 , xdot, libcdt5, libpathplan4, libgvpr2, libgvc6, graphviz

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("iris")