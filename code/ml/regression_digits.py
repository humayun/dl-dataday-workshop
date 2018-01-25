

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns






digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_data = data[:n_samples//2]
test_data = data[n_samples//2:]
y_train = digits.target[:n_samples//2]
y_test = digits.target[n_samples//2:]


# regr = linear_model.LinearRegression()
logisticRegr = LogisticRegression()

# Train the model using the training sets
logisticRegr.fit(train_data, y_train)

# Make predictions using the testing set

score = logisticRegr.score(test_data, y_test)
print(score)


from sklearn import metrics
predictions = logisticRegr.predict(test_data)
cm = metrics.confusion_matrix(y_test, predictions)


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
# plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
plt.show();






