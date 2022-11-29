import sys
import numpy as np
import pandas as pd
import math
import time

args = sys.argv

input_path = args[1]
output_path = args[2]

# Loading Datset

train = pd.read_csv(input_path + 'train_data_shuffled.csv', header=None)

cols = train.columns

X_train = train.drop(columns=cols[-1])
Y_train = train[cols[-1]]

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

X_train = X_train/255.0

input_layer = X_train.shape[1]

class Neural_Network:
    
    def __init__(self, input_layer, layers):
        
        L = len(layers)
        np.random.seed(1)
        
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
        self.t = 1
        
        change_w = [None for i in range(L)]
        change_b = [None for i in range(L)]
        dw_sq = [None for i in range(L)]
        db_sq = [None for i in range(L)]
        
        for i in range(L):
            change_w[i] = np.zeros(weights[i].shape)
            change_b[i] = np.zeros(bias[i].shape)
            dw_sq[i] = np.zeros(weights[i].shape)
            db_sq[i] = np.zeros(bias[i].shape)
            
        self.change_w = change_w
        self.change_b = change_b
        self.dw_sq = dw_sq
        self.db_sq = db_sq
            
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
    
    def backward(self, X, Y, lr, activation, lr_type, t, loss_type, sgd, beta1, beta2):
        
        if lr_type == 1:
            lr = lr/np.sqrt(t)
        
        weights = self.weights
        bias = self.bias
        outputs = self.outputs
        L = self.L
        
        m = X.shape[0]
        
        grads_w = [None for i in range(L)]
        grads_b = [None for i in range(L)]
        
        if sgd == 1:
            weights = future_params(weights, self.change_w, beta1)
            bias = future_params(bias, self.change_b, beta1)
        
        if loss_type == 0:
            delta = outputs[L-1] - Y
        else:
            delta = (outputs[L-1] - Y)*derivative(outputs[L-1], activation)
            
        grads_w[L-1] = np.dot(outputs[L-2].T, delta)/m
        grads_b[L-1] = np.sum(delta, axis = 0)/m
        
        for i in range(L-2,-1,-1):
            
            delta = derivative(outputs[i], activation)*np.dot(delta, weights[i+1].T)
            
            if i == 0:
                grads_w[i] = np.dot(X.T, delta)/m
            else:
                grads_w[i] = np.dot(outputs[i-1].T, delta)/m
                
            grads_b[i] = np.sum(delta, axis = 0)/m
            

        self.change_w = momentum(self.change_w, grads_w, lr, beta1)
        self.change_b = momentum(self.change_b, grads_b, lr, beta1)
        
        self.dw_sq = rmsprop(self.dw_sq, grads_w, beta2)
        self.db_sq = rmsprop(self.db_sq, grads_b, beta2)
            
        for i in range(L):
            if sgd == 0 or sgd == 1:
                self.weights[i] -= self.change_w[i]
                self.bias[i] -= self.change_b[i]
                
            elif sgd == 2:
                self.weights[i] -= lr*grads_w[i]*(1.0/np.sqrt(self.dw_sq[i]+math.pow(10,-8)))
                self.bias[i] -= lr*grads_b[i]*(1.0/np.sqrt(self.db_sq[i]+math.pow(10,-8)))
                
            elif sgd == 3 or sgd == 4:
                mw_ = self.change_w[i]/(1-math.pow(beta1,self.t))
                mb_ = self.change_b[i]/(1-math.pow(beta1,self.t))
                
                vw_ = self.dw_sq[i]/(1-math.pow(beta2,self.t))
                vb_ = self.db_sq[i]/(1-math.pow(beta2,self.t))
                
                if sgd == 4:
                    mw_ = beta1*mw_ + (1-beta1)*grads_w[i]/(1-math.pow(beta1, self.t))
                    mb_ = beta1*mb_ + (1-beta1)*grads_b[i]/(1-math.pow(beta1, self.t))
                    
                self.weights[i] -= lr*mw_*(1.0/(np.sqrt(vw_)+math.pow(10,-8)))
                self.bias[i] -= lr*mb_*(1.0/(np.sqrt(vb_)+math.pow(10,-8)))
                    
                self.t += 1  
    
            else:
                self.weights[i] -= lr*grads_w[i]
                self.bias[i] -= lr*grads_b[i]

def rmsprop(grad_sq, grads, beta2):
    
    for i in range(len(grads)):
        grad_sq[i] = beta2*grad_sq[i] + (1-beta2)*np.square(grads[i])
    return grad_sq

def future_params(weights, change, beta1):
    for i in range(len(weights)):
        weights[i] = weights[i] - beta1*change[i]
        
    return weights

def momentum(change, grads, lr, beta1):
    
    for i in range(len(grads)):
        change[i] = beta1*change[i] + lr*grads[i]
        
    return change

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
        return -1*np.sum(y_test*np.log(y_pred))/m
    else:
        return (1/2) * np.sum(np.square(y_test-y_pred))/m

def one_hot(y, classes):
    m = y.shape[0]
    y_hot = np.zeros((m, classes))
    y_hot[np.arange(m), y] = 1
    
    return y_hot

lr = 0.1
activation = 2
lr_type = 1
loss_type = 0
batch_size = 256
sgd = 0
beta1 = 0.9
beta2 = 0.999
layers = [512,256,46]

def training(X, Y, model, lr, activation, lr_type, loss_type, batch_size, sgd, beta1, beta2):

    start = time.time()
    
    training_loss = []
    classes = len(np.unique(Y))
    
    Y = one_hot(Y, classes)
    m = X.shape[0]
    
    i = 0
    while time.time()-start < 260:
        for j in range(0,m,batch_size):
            
            if j+batch_size>m:
                break

            X_batch = X[j:j+batch_size, :]
            Y_batch = Y[j:j+batch_size, :]

            model.forward(X_batch, activation, loss_type)
            model.backward(X_batch, Y_batch, lr, activation, lr_type, i+1, loss_type, sgd, beta1, beta2)
        
        i += 1

        if i == 29:
            break

    return i

model = Neural_Network(input_layer = input_layer, layers=layers)
epochs = training(X_train, Y_train, model, lr, activation, lr_type, loss_type, batch_size, sgd, beta1, beta2)

for i in range(len(layers)):
    b = model.bias[i].reshape((1,-1))
    temp = np.concatenate((b, model.weights[i]), axis = 0)
    np.save(output_path+'w_'+str(i+1)+'.npy', temp)

my_params = [epochs, 256, 1, 0.1, 2, 0, 1, layers, 1]

with open(output_path+'my_params.txt','w') as file:
    for p in my_params:
        file.write(str(p)+'\n')
