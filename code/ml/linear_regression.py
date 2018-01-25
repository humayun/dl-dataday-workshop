import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


boston = datasets.load_boston()  ## housing prices in boston using these 13 features
# print(boston.keys())
# print(boston.data.shape)
# print(boston.feature_names)
# print(boston.DESCR)
# print(boston.target)

feature_no = 0
boston_X = boston.data[:, np.newaxis, feature_no ] # use the crime rate feature /first feature

plt.scatter(boston_X,boston.target)
plt.xlabel("crime rate")
plt.ylabel("house price")
plt.title("relationship between crime rate and housing price")
plt.show()




# let's split the data into train and test
length = len(boston.data)
train_ratio = 0.8
tr_r = int(train_ratio * length)

boston_X_train = boston_X[:tr_r]
boston_X_test = boston_X[tr_r:]
boston_y_train = boston.target[:tr_r]
boston_y_test = boston.target[tr_r:]

# Create linear regression object
regr = linear_model.LinearRegression()
# let's train the model
regr.fit(boston_X_train, boston_y_train)

boston_y_pred = regr.predict(boston_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(boston_y_test, boston_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(boston_y_test, boston_y_pred))


# let's plot the line!
print(boston_X_test.shape)
print(boston_y_test.shape)
plt.scatter(boston_X_test, boston_y_test,  color='black')
plt.plot(boston_X_test, boston_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

