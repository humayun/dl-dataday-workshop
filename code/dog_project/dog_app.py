import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import ImageFile
from sklearn.datasets import load_files

from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization


import random
random.seed(8675309)


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
def Count_Faces(Image_Sets):
    image_names = []
    for image_name in Image_Sets:
        if face_detector(image_name):
            image_names.append(image_name)
    return len(image_names), np.array(image_names)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
def Count_Dog(Image_Sets):
    image_names = []
    for image_name in Image_Sets:
        if dog_detector(image_name):
            image_names.append(image_name)
    return len(image_names), np.array(image_names)



# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
no_dog_breeds = len(dog_names)


# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

total_faces, human_files_selected = Count_Faces(human_files_short)
total_dogs, dog_files_selected = Count_Faces(dog_files_short)

print('Percentage of Images (Human Images) which have Human Face is: ' + str(total_faces/100.*100)+'%')
print('Percentage of Images (Dog Images) which have Human Face is: ' + str(total_dogs/100.*100)+'%')

### Step 2: Detect Dogs

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

total_dogs_1, human_files_selected_2 = Count_Dog(human_files_short)
total_dogs_2, dog_files_selected_1 = Count_Dog(dog_files_short)

print('Percentage of Images (Human Images) which have Dog face is: ' + str(total_dogs_1)+'%')
print('Percentage of Images (Dog Images) which have Dog face is: ' + str(total_dogs_2)+'%')

ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


K.clear_session()
model = Sequential()

### TODO: Define your architecture.
model.add(BatchNormalization(input_shape=(224,224,3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())

model.add(Dense(no_dog_breeds, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 50
batch_size = 20

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')

# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


