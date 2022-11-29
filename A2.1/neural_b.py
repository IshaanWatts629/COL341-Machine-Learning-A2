import sys
import numpy as np
import pandas as pd

args = sys.argv

input_path = args[1]
output_path = args[2]
params = args[3]

# Reading Parameters

with open(params, 'r') as file:
    data = np.loadtxt(file, dtype = 'str')

    epochs = int(data[0])
    batch_size = int(data[1])
    layers = [int(i) for i in data[2][1:-1].split(',')]
    lr_type = int(data[3])
    lr = float(data[4])
    activation = int(data[5])
    loss_fxn = int(data[6])
    seed_value = int(data[7])

# Loading Datset

train = pd.read_csv(input_path + 'train_data_shuffled.csv', header=None)
test = pd.read_csv(input_path + 'public_test.csv', header=None)

cols = train.columns

X_train = train.drop(columns=cols[-1])
Y_train = train[cols[-1]]

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

X_train = X_train/255.0

X_test = test.drop(columns=cols[-1])
Y_test = test[cols[-1]]

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

X_test = X_test/255.0

input_layer = X_train.shape[1]

# Model

class Neural_Network:

    def __init__(self, input_layer, layers):

        L = len(layers)
        np.random.seed(seed_value)
        print(seed_value)

        weights = [None for i in range(L)]
        outputs = [None for i in range(L)]
        bias = [None for i in range(L)]

        params = np.random.normal(0, 1, size = (input_layer+1, layers[0])).astype('f') * np.sqrt(2/(input_layer+1+layers[0]))
        bias[0] = params[0,:]
        weights[0] = params[1:,:]

        for i in range(1,L):
            params = np.random.normal(0,1, size = (layers[i-1]+1, layers[i])).astype('f') * np.sqrt(2/(layers[i-1]+1+layers[i])) 
            bias[i] = params[0,:]
            weights[i] = params[1:,:]

        self.weights = weights
        self.bias = bias
        self.L = L
        self.outputs = outputs

    def forward(self, X, activation, loss_type):

        weights = self.weights
        outputs = self.outputs
        bias = self.bias
        L = self.L
        for i in range(L-1):
            if i == 0:
                Z = np.dot(X, weights[0]) + bias[0]
            else:
                Z = np.dot(outputs[i-1], weights[i]) + bias[i]
            self.outputs[i] = activation_fxn(Z, activation)

        Z = np.dot(outputs[L-2], weights[L-1]) + bias[L-1]

        if loss_type == 0:
            self.outputs[L-1] = softmax(Z)
        else:
            self.outputs[L-1] = activation_fxn(Z, activation)

        return outputs[L-1]   

    def predict(self, X, activation, loss_type):

        y_pred = self.forward(X, activation, loss_type)
        return np.argmax(y_pred, axis = 1)

    def backward(self, X, Y, learning_rate, activation, lr_type, t, loss_type):
        
        if lr_type == 1:
            learning_rate = learning_rate/np.sqrt(t)

        weights = self.weights
        bias = self.bias
        outputs = self.outputs
        L = self.L
        m = X.shape[0]

        grads_w = [None for i in range(L)]
        grads_b = [None for i in range(L)]

        if loss_type == 0:
            delta = outputs[L-1] - Y
        else:
            delta = (outputs[L-1] - Y)*derivative(outputs[L-1], activation)

        grads_w[L-1] = np.dot(outputs[L-2].T, delta)
        grads_b[L-1] = np.sum(delta, axis = 0)

        for i in range(L-2,-1,-1):
            delta = derivative(outputs[i], activation)*np.dot(delta, weights[i+1].T)

            if i == 0:
                grads_w[i] = np.dot(X.T, delta)
            else:
                grads_w[i] = np.dot(outputs[i-1].T, delta)

            grads_b[i] = np.sum(delta, axis = 0)

        for i in range(L):
            self.weights[i] -= learning_rate*grads_w[i]*(1/m)
            self.bias[i] -= learning_rate*grads_b[i]*(1/m)

def softmax(X):
    X = X - np.max(X, axis = 1, keepdims=True)
    A = np.exp(X)
    ans = A/np.sum(A, axis = 1, keepdims = True)
    return ans

def sigmoid(X):
    return 1.0/(1.0 + np.exp(-1.0*X))

def relu(X):
    X[X<0] = 0
    return X 

def activation_fxn(X, activation):
    if activation == 0:
        return sigmoid(X)
    elif activation == 1:
        return np.tanh(X)
    else:
        return relu(X)

def derivative(X, activation):
    if activation == 0:
        return X*(1-X)
    elif activation == 1:
        return 1-np.square(X)
    else:
        X[X<0] = 0
        X[X>0] = 1
        return X

def loss(y_test, y_pred, loss_type):

    m = y_test.shape[0]

    if loss_type == 0:
        return -1*np.sum(y_test*np.log(y_pred))*(1/m)
    else:
        return (1/2)*np.sum(np.square(y_test-y_pred))*(1/m)

def one_hot(y, classes):
    m = y.shape[0]
    y_hot = np.zeros((m, classes))
    y_hot[np.arange(m), y] = 1
    
    return y_hot

def training(X, Y, model, epochs, lr, activation, lr_type, loss_type, batch_size):
    
    training_loss = []
    Y = one_hot(Y, len(np.unique(Y)))
    m = X.shape[0]
    
    for i in range(epochs):
        for j in range(0,m,batch_size):
            if j+batch_size>m:
                break

            X_batch = X[j:j+batch_size, :]
            Y_batch = Y[j:j+batch_size, :]

            model.forward(X_batch, activation, loss_type)
            model.backward(X_batch, Y_batch, lr, activation, lr_type, i+1, loss_type)

        Y_ = model.forward(X, activation, loss_type)
        l = loss(Y, Y_, loss_type)
        training_loss.append(l)

    return training_loss

# Training

model = Neural_Network(input_layer = input_layer, layers=layers)
error = training(X_train, Y_train, model, epochs, lr, activation, lr_type, loss_fxn, batch_size)

result = model.predict(X_test, activation, loss_fxn)
np.save(output_path+'predictions.npy', result)

for i in range(len(layers)):
    b = model.bias[i].reshape((1,-1))
    temp = np.concatenate((b, model.weights[i]), axis = 0)
    np.save(output_path+'w_'+str(i+1)+'.npy', temp)
