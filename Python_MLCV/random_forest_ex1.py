import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_img, train_lbl = train_set
test_img, test_lbl = test_set


"""
# Random Forest Classifier with default parameters 
from sklearn.ensemble import RandomForestClassifier
X = train_img[0:999]
y = train_lbl[0:999]
clf = RandomForestClassifier()
clf = clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy: ", acc) # 0.7912

"""

# now testing parameter sets
from sklearn.ensemble import RandomForestClassifier
X = train_img[0:999]
y = train_lbl[0:999]
clf = RandomForestClassifier(criterion = 'gini', n_estimators = 100, max_depth = 15, max_features = 10)
clf = clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy: ", acc) # 0.7912

"""
    testing with 1000 test data
    gini                                              acc = 0.7912
    entropy                                           acc = 0.7880
    gini, n_estimators = 100                          acc = 0.8863
    gini, n_estimators = 1000                         acc = 0.8925
    gini, n_estimators = 1000, max_depth 10           acc = 0.8929
    gini, n_esti 500, max_depth 10, max_feat 50       acc = 0.8855
    gini, n_esti 2000, max_depth 15, max_feat 100     acc = 0.8828
    gini, n_esti 1000, max_depth 15, max_feat 10      acc = 0.8954
"""


# visualize the pixel importances
pixel_imp= clf.feature_importances_
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.imshow(pixel_imp.reshape((28,28)), cmap = cm.Greys_r)
plt.show()



# now do cross-validation:
# takes too long to compute
"""
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_img, train_lbl, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = RandomForestClassifier(criterion = 'gini', n_estimators = 1000, max_depth = 15, max_features = 10)
clf = clf.fit(X_train,y_train)
print(clf.score(X_test, y_test)) 


# mean score of cross-validation
from sklearn import cross_validation
clf = RandomForestClassifier(criterion = 'gini', n_estimators = 1000, max_depth = 15, max_features = 10)
scores = cross_validation.cross_val_score(clf, train_img, train_lbl, cv=5)
print(scores)
"""


