from sklearn import tree

# 0 = bumpy, apple
# features are weight and shape

features = [[140, 1],[130, 1],[150, 0],[170, 0]]

labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

print(clf.predict([[150, 0]]))
