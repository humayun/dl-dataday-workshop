from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = load_model("model.h5")

pred = model.predict(X_test)

digit = X_test[1]

str = ""
for i in range(digit.shape[0]):
    for j in range(digit.shape[1]):
        if digit[i][j] == 0:
            str += " "
        elif digit[i][j] < 128:
            str += "."
        else:
            str += "X"
    str += "\n"

print(str)
for i in range(len(pred[1])):
    print("\n%d: %.2f%%" % (i, pred[1][i]))
print(pred[1])
