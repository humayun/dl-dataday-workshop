from keras.datasets import mnist
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = [x.flatten() for x in X_train]

perceptron = Perceptron()

scores = cross_val_score(perceptron, X_train, y_train, cv=10)

print(scores)
print(scores.mean())

