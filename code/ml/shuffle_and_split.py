import numpy as np 
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()

length = len(digits.images)
print(length)


rand = np.random.RandomState()
shuffle = rand.permutation(length)

images , labels = digits.images[shuffle], digits.target[shuffle]

test_ratio = 0.2
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
print("size of training data:")
print(len(train_images))
print("size of evaluation data:")
print(len(eval_images))
print("size of test data:")
print(len(test_images))

# print(len(train_images)+len(eval_images)+len(test_images))

# display the first image in train
plt.title('Label is {label}'.format(label=train_labels[0]))
plt.imshow(train_images[0], cmap='gray')
plt.show()





