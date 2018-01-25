

from keras.datasets import mnist #mnist dataset is included in Keras datasets library
import matplotlib.pyplot as plt



(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# size of training data
print("Training data shape:")
print(X_train.shape)
#print(X_train.shape[0])

print("Test data shape:")
print(X_test.shape)

#plot images
sample = 300 # put the image number you want to see here
plt.title('Label is {label}'.format(label=y_train[sample]))
plt.imshow(X_train[sample], cmap='gray')
plt.show()


