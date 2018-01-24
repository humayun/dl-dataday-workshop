from numpy import genfromtxt

# Read a text (csv) file
def Load_Data(file_name):
    data = genfromtxt(file_name, delimiter=',')
    Y = data[:,-1]
    X = data[:,0:-1]
    return X, Y

def Predict(X, W, B):
    score = 0.
    # Score = WX + B
    for i in range(len(X)):
        score += W[i] * X[i]
    score += B
    if score >= 0.:
        return 1.
    return 0.

def Train_Perceptron(X, Y, W, B, learning_rate):
    for i in range(len(X)):
        # Call Predict function by passing W, B and X to compute the score
        prediction = Predict(X[i], W, B)
        error = Y[i] - prediction
        # Adjust weights and Bias
        # W'[i] =  W[i] + (LR * Error * X[i])
        for j in range(len(W)):
            W[j] = W[j] +  (learning_rate * error * X[i,j])
        # For Bias
        # B = B + (LR * Error)
        B = B + learning_rate * error
    return W, B

def Evaluate_Perceptron(X, Y, W, B):
    correct = 0.
    for i in range(len(X)):
        if Predict( X[i], W, B) == Y[i]:
            correct += 1.
    return correct / float(len(Y)) * 100.


file_name = 'data_1.csv'
X, Y = Load_Data(file_name)

# initialize weights and bias as 0
W = [0. for i in range(len(X[0]))] # number weights parameters is equal to number of features (len(X[0])
B = 0.
learning_rate = 0.1
n_epoch = 100

for epoch in range(n_epoch):
    # Train the Perceptron
    W, B = Train_Perceptron(X, Y, W, B, learning_rate )

    # Evaluate the Perceptron
    accuracy = Evaluate_Perceptron(X, Y, W, B)
    print('Epoch: {0} \tMain Accuracy: {1} %.'.format(epoch+1,accuracy))
