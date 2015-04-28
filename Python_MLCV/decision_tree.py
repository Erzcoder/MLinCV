import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_img, train_lbl = train_set
test_img, test_lbl = test_set

"""
# using a decision tree classifier with default parameters
from sklearn import tree
X = train_img[0:999]
y = train_lbl[0:999]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy: ", acc) # 0.6812 - way too low
"""

# Now change the criterion to entropy (only with 1000 training examples)
from sklearn import tree
X = train_img[0:999]
y = train_lbl[0:999]
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 20, max_features = 50)
clf = clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy: ", acc) # 0.6599 - even a little bit lower

"""
    gini                                       acc = 0.6812
    entropy                                    acc = 0.6599
    gini, mdepth = 10                          acc = 0.6875
    gini, mdepth = 100                         acc = 0.6772
    entropy, mdepth = 10                       acc = 0.6653
    entropy, mdepth = 1000                     acc = 0.6614
    gini, mdepth = 1000                        acc = 0.6924
    gini, mdepth = 1000, mfeatures = 10        acc = 0.5688
    gini, mdepth = 1000, mfeatures = 100       acc = 0.6097
    gini, mdepth 5, mfeat 5                    acc = 0.4242
    gini, mdepth 5, mfeat 50                   acc = 0.5255
    entropy, mdepth 2, mfeat 100               acc = 0.3270
    entropy, mdepth 20, mfeat 50               acc = 0.6215

    Now testing with 10.000 training points
    entropy, mdepth 20, mfeatures 50           acc = 0.7957

    Now testing with all training points
    entropy, mdepth 20, mfeatures 50           acc = 0.8525
    
    
"""
"""
# visualize the constructed tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file= dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("mnist_tree2.pdf")
"""


# visualize the pixel importances
pixel_imp= clf.feature_importances_
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.imshow(pixel_imp.reshape((28,28)), cmap = cm.Greys_r)
plt.show()


# now do cross-validation

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_img, train_lbl, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 20, max_features = 50)
clf = clf.fit(X_train,y_train)
print(clf.score(X_test, y_test)) # 0.8281

# mean score of cross-validation
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 20, max_features = 50)
scores = cross_validation.cross_val_score(clf, train_img, train_lbl, cv=5)
print(scores)

