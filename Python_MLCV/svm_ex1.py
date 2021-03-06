import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_img, train_lbl = train_set
test_img, test_lbl = test_set

    
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# plt.imshow(train_img[5].reshape((28,28)), cmap = cm.Greys_r)
# plt.imshow(test_img[5].reshape((28,28)), cmap = cm.Greys_r)
plt.show()
print(train_img.shape)# dimensions: (50.000, 784)
print(test_img.shape) # dimensions: (10.000, 784)
print(test_lbl[5])  # label is 1, as show in the picture
"""

# Now to Exercise 2
"""
from sklearn import svm
X = train_img[0:999]
y = train_lbl[0:999]
clf = svm.SVC() # default kernel is rbf, SVC is Support Vector Classifier
clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy, rbf kernel: ", acc) # 0.826 - way too low!
"""

# Now with tuning of hyperparameters
"""
# test with linear kernel
from sklearn import svm
X = train_img[0:999]
y = train_lbl[0:999]
clf = svm.SVC(kernel="linear")
clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy, linear kernel: ", acc) # 0.876 - still way too low!
"""

"""
    test with poly kernel:          accuracy 0.1028
    test with sigmoid kernel:       accuracy 0.1028
    """

# test with linear kernel
from sklearn import svm
X = train_img[0:999]
y = train_lbl[0:999]
clf = svm.SVC(kernel="rbf", gamma = 0.01, C = 5)
clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_img[0:5]))
print("The image's actual label ist: ")
print(test_lbl[0:5])

pred = clf.predict(test_img)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_lbl)
print("accuracy: ", acc) # 0.9132

"""
    test with 1000 training points
    test with rbf kernel and gamma = 1: acc = 0.1174
    gamma = 10 --> acc = 0.1028
    gamma = 0.1 --> acc = 0.6158
    gamma = 0.01 --> acc = 0.9024
    gamma = 0.001 --> acc = 0.807

    test with rbf kernel, gamma = 0.01 and different C values:
    C = 1 --> acc = 0.9024
    C = 10 --> acc = 0.9125
    C = 100 --> acc = 0.9125
    C = 0.1 --> acc = 0.77
    C = 20 --> acc = 0.9125
    C = 5 --> acc = 0.9132
    C = 2 --> acc = 0.9105

    now C = 5 and gamma = 1 --> acc = 0.1228

    So, the best set of parameters found was: kernel = rbf, gamma = 0.01, C = 5

  -----------------------------------------------
    Now, test with 10.000 training data:

    C = 5 and gamma = 0.01 --> acc = 0.9648

    And now, finally test with ALL training data:
    
    C = 5 and gamma = 0.01 --> acc = 0.982
    
    
    
"""



