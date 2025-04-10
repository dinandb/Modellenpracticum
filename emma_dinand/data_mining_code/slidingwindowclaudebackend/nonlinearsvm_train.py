

from sklearn.svm import SVC


def train(X, y):
    svm_clf = SVC(kernel='rbf', gamma='scale', C=1)
    svm_clf.fit(X, y)

    return svm_clf