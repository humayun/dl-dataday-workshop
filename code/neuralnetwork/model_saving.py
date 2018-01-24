from keras.datasets import mnist
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='zeros'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=200, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('model.h5', overwrite=True)

# later...

# load model
loaded_model = load_model('model.h5')
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

