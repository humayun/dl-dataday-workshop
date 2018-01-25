
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt



#let's load digits data

digits = datasets.load_digits()
# print(digits.target.shape)
# print(digits.images.shape)
# exit()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) #reshape data - flatten images to input to classifire
train_data = data[:n_samples//2] # take 50% of data for training
test_data = data[n_samples//2:]
y_train = digits.target[:n_samples//2]
y_test = digits.target[n_samples//2:]







# let's train the model
classifier = svm.SVC(C = 1,gamma=0.001)
classifier.fit(train_data, y_train)

# let's predict


predicted = classifier.predict(test_data)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))

#how to save a trained model and load it later

# import pickle
# s = pickle.dumps(classifier)
# model = pickle.loads(s)
# print("Reloading the model for prediction")
# print(test_data.shape)
# predictions = model.predict(test_data)
# print("Classification report after reloading the model %s:\n%s\n"
#       % (classifier, metrics.classification_report(y_test, predictions)))




