from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_width = X_train.shape[1]
img_height = X_train.shape[2]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# build model

model = Sequential()
model.add(Conv2D(6, (3, 3), input_shape=(img_width, img_height,1), activation='relu', padding="same")) # padding='valid'
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test))
#model.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test),callbacks=[tensorboard, WandbKerasCallback()])

model.save('model.h5')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
