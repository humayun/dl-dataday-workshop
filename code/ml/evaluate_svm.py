import numpy as np 
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()

length = len(digits.images)
print(length)


rand = np.random.RandomState()
shuffle = rand.permutation(length)

images , labels = digits.images[shuffle], digits.target[shuffle]

test_ratio = 0.5
eval_ratio = 0.3

tst_r = int(test_ratio * length)
# print(length)
# print(tst_r)

test_images = images[:tst_r] #let's split data into test and train 20/80
train_images = images[tst_r:]

test_labels = labels[:tst_r]
train_labels = labels[tst_r:]
# print(len(test_images))
# print(len(train_images))
# print(len(train_images)+len(test_images))

eval_r = int(eval_ratio * len(train_images)) #let's split train data to training samples and evaluation samples 70/30
eval_images = train_images[:eval_r]
train_images = train_images[eval_r:]

eval_labels = train_labels[:eval_r]
train_labels = train_labels[eval_r:]



# convert to right input format for training model 
def reshape_data(data):
	n_samples = len(data)
	data = data.reshape((n_samples, -1))
	return data

train_images = reshape_data(train_images)
eval_images = reshape_data (eval_images)
test_images = reshape_data (test_images)


# train and evaluate the model
classifier = svm.SVC(C = 0.1,gamma=0.001)
classifier.fit(train_images, train_labels)

predicted = classifier.predict(eval_images)

from sklearn.metrics import accuracy_score
valid_score = accuracy_score(eval_labels, predicted)
print("validation accuracy:")
print(valid_score)

# test the model
predicted = classifier.predict(test_images)
test_score = accuracy_score(test_labels, predicted)
print("test accuracy:")
print(test_score)







