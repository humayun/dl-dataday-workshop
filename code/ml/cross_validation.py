
from sklearn import datasets,svm
from sklearn.model_selection import cross_val_score

digits = datasets.load_digits()

model = svm.SVC(C = 0.1,gamma=0.001)

scores = cross_val_score(model, digits.data, digits.target, cv=10)
print(scores)
print(scores.mean())