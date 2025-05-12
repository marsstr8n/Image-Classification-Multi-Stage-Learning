# goal: create a multi-layered perceptron from scratch

# input: image vector of 3072 numbers, [1, 3072]

# 2 hidden layers - using reLu, 512 hidden to 128 hidden. Each weight matrix is weights x neurons: [3072, 512] and [512, 128]
# output layer: [128,1]

# output: one neuron - use sigmoid activation function
#      OR 2 neurons and use softmax

# this is the order of MLP: image input -> hidden layer 1 -> output of HL1 -> reLU 1 -> HL2 -> output of HL2 -> reLU 2 -> output layer (1 neuron) -> sigmoid (ouput) to get predicted values.

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# load the preprocessed_data.npz
loaded = np.load("preprocessed_data.npz", allow_pickle=True)
imagedata = loaded['data'] # shape is (24946, 3072) - each image taking the row
labels = loaded['labels'] # 0, 1
class_names = loaded['class_names'] # 'Cat', 'Dog'


# =======FIRST ITERATION OF THE MLP CLASS FROM SCRATCH=======
# class MLP:
#     def __init__(self, layer_sizes: list, activation='relu', learn_rate=0.01):
#         self.layer_sizes = layer_sizes # list representing the number of neurons in each lauyer - from input, to output e.g. [3072, ..., 1]
#         self.activation = activation
#         self.learn_rate = learn_rate
#         self.weights = [] # store all the weight matrices of all layers
#         self.biases = []
#         self.initialise_weights()
    
#     @staticmethod    
#     def relu(x: np.ndarray) -> np.ndarray: 
#         """Apply ReLU activation function to the output of the forward pass. This changes any negative value to 0, keep positive values as is

#         Args:
#             x (np.ndarray): dot product value, after going through a layer
#         """
#         return np.maximum(0, x) # first arg is scalar value 0, it compares each value in x to it
#                             # thus replaces any negative value (lower than 0) and keep positive values
    
#     @staticmethod                          
#     def sigmoid(x: np.ndarray) -> np.ndarray:
#         """to be applied at the output layer

#         Args:
#             x (np.ndarray): [number of images, 1]

#         Returns:
#             np.ndarray: [number of images, 1] - values between 0 and 1
#         """
#         return 1 / (1 + np.exp(-x))

#     @staticmethod            
#     def relu_derivative(x): 
#         """derivative of relu - the activation we used between hidden layers
#         x: ouput of a hidden layer
#         relu = 1 if x > 0 -> d/dx(relu) = 1 when x > 0
#         relu = 0 if x < 0 -> d/dx(relu) = 0 when x > 0

#         thus, this function returns True or False, with changing the Boolean to float - which gives True = 1, False = 0
#         Returns an array of 1s and 0s
#         """
#         return (x > 0).astype(float)

#     @staticmethod    
#     def sigmoid_derivative(x):
#         sig = 1 / (1 + np.exp(-x))
#         return sig * (1 - sig)

    
#     def initialise_weights(self):
#         # initialise weights and biases - for each layer input
#         for i in range(1, len(self.layer_sizes)):
#             # create a matrix of weights between layer i-1 and i / use He initialisation
#             weight_matrix = np.random.rand(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2. /self.layer_sizes[i-1]) # i-1:previous layer 
#             bias_matrix = np.zeros((1, self.layer_sizes[i]))
#             self.weights.append(weight_matrix) # store each intialised weight matrix into the list
#             self.biases.append(bias_matrix)
    
#     def forward_pass(self, X) -> np.ndarray:
#         """this function calculates the dot product of input vector, weight matrix of a hidden layer
#         in a form of [weights, neuron] + the bias weights [1, neuron_amount]

#         Input: the first input - this is the original image vector 

#         Output: 
#             z: a list of output matrix of each layer
#             a: activation function using z matrix of that layer
            
#         """
#         self.z_list = [] 
#         self.activations_list = [X] # take the first input as activation for input layer
        
#         for i in range(len(self.weights)):
#             z = np.dot(self.activations_list[-1], self.weights[i]) + self.biases[i] # represent z = input @ weight_matrix + b
#             self.z_list.append(z)
            
#             if i == len(self.weights) - 1: # last layer
#                 a = self.sigmoid(z)
#             else:
#                 a = self.relu(z) # relu in between
                
#             self.activations_list.append(a)
#         return self.activations_list[-1] # returns final output (y_hat)
    
#     def compute_loss(self, y_true, y_pred):
#         # Binary Cross-Entropy Loss
#         eps = 1e-8
#         y_pred = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
#         return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    
#     def backpropagation(self, y_true):
#         y_pred = self.activations_list[-1]
#         learning_rate = self.learn_rate

#         # STEP 1: OUTPUT LAYER
#         dL_dz = y_pred - y_true  # shape: (batch_size, output_size)

#         a_prev = self.activations_list[-2]  # activation before output layer
#         dL_dw = a_prev.T @ dL_dz
#         dL_db = np.sum(dL_dz, axis=0, keepdims=True)

#         self.weights[-1] -= self.learn_rate * dL_dw
#         self.biases[-1] -= self.learn_rate * dL_db

#         # STEP 2: BACKPROPAGATE THROUGH HIDDEN LAYERS
#         for i in reversed(range(len(self.weights) - 1)):
#             z = self.z_list[i]
#             a_prev = self.activations_list[i]

#             # Derivative of activation
#             if self.activation == 'relu':
#                 grad_act = self.relu_derivative(z)  # use z[i] here
#             else:
#                 raise NotImplementedError("Only relu is supported for hidden layers.")

#             # Backpropagate the gradient
#             dL_dz = (dL_dz @ self.weights[i + 1].T) * grad_act

#             dL_dw = a_prev.T @ dL_dz
#             dL_db = np.sum(dL_dz, axis=0, keepdims=True)

#             self.weights[i] -= self.learn_rate * dL_dw
#             self.biases[i] -= self.learn_rate * dL_db
        
#     def train_on_batch(self, X, y):
#         y_pred = self.forward_pass(X)
#         self.backpropagation(y)
#         loss = self.compute_loss(y, y_pred)
#         return loss     
    
#     def predict(self, X):
#         y_pred = self.forward_pass(X)
#         return (y_pred > 0.5).astype(int)
    
#     def evaluate(self, X, y_true):
#         y_pred = self.predict(X)
#         accuracy = np.mean(y_pred == y_true)
#         return accuracy
        
# =======SECOND ITERATION OF THE MLP CLASS FROM SCRATCH=======
class MLP: # no dropout, no optimiser (e.g. Adam)
    def __init__(self, layer_sizes: list, activation='relu', learn_rate=0.001):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learn_rate = learn_rate
        self.weights = []
        self.biases = []
        self.initialise_weights()

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid_derivative(x):
        x = np.clip(x, -500, 500)
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)

    def initialise_weights(self):
        for i in range(1, len(self.layer_sizes)):
            weight_matrix = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) * np.sqrt(2. / self.layer_sizes[i - 1])
            bias_matrix = np.zeros((1, self.layer_sizes[i]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

    def forward_pass(self, X):
        z_list = []
        activations_list = [X]

        for i in range(len(self.weights)):
            z = np.dot(activations_list[-1], self.weights[i]) + self.biases[i]
            z_list.append(z)

            if i == len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)

            activations_list.append(a)

        return activations_list[-1], activations_list, z_list

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_true = y_true.reshape(y_pred.shape)  # ensure shapes match
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backpropagation(self, y_true, y_pred, activations_list, z_list):
        dL_dz = y_pred - y_true  # Gradients of the loss with respect to z at the output layer

        a_prev = activations_list[-2]  # Activation from the previous layer
        dL_dw = a_prev.T @ dL_dz  # Gradient w.r.t. weights (dot product)
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)  # Gradient w.r.t. biases (sum over batch axis)

        # Gradient clipping
        max_grad = 5.0
        dL_dw = np.clip(dL_dw, -max_grad, max_grad)
        dL_db = np.clip(dL_db, -max_grad, max_grad)

        # Update weights and biases for the last layer
        self.weights[-1] -= self.learn_rate * dL_dw
        self.biases[-1] -= self.learn_rate * dL_db

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            z = z_list[i]
            a_prev = activations_list[i]

            if self.activation == 'relu':
                grad_act = self.relu_derivative(z)
            else:
                raise NotImplementedError("Only ReLU is supported for hidden layers.")

            dL_dz = (dL_dz @ self.weights[i + 1].T) * grad_act  # Propagate the error backward

            dL_dw = a_prev.T @ dL_dz  # Gradient w.r.t. weights for hidden layers
            dL_db = np.sum(dL_dz, axis=0, keepdims=True)  # Sum over batch axis for bias gradients

            # Gradient clipping
            dL_dw = np.clip(dL_dw, -max_grad, max_grad)
            dL_db = np.clip(dL_db, -max_grad, max_grad)

            # Update weights and biases for hidden layers
            self.weights[i] -= self.learn_rate * dL_dw
            self.biases[i] -= self.learn_rate * dL_db
            
    def train_on_batch(self, X, y):
        y_pred, activations_list, z_list = self.forward_pass(X)
        self.backpropagation(y, y_pred, activations_list, z_list)
        loss = self.compute_loss(y, y_pred)
        return loss

    def predict(self, X):
        y_pred, _, _ = self.forward_pass(X)
        return (y_pred > 0.5).astype(int)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

# =======THIRD ITERATION OF THE MLP CLASS FROM SCRATCH=======
class MLP3: # dropout, optimiser (e.g. Adam) included
    def __init__(self, layer_sizes: list, activation='relu', dropout = 0.0, learn_rate=0.001):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learn_rate = learn_rate
        self.dropout = dropout
        self.weights = []
        self.biases = []
        self.initialise_weights()
        self.t = 1 # timestep for Adam
        
        # for Adam optimiser
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]



    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid_derivative(x):
        x = np.clip(x, -500, 500)
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    
    # dropout function
    def apply_dropout(self, a, rate):
        mask = (np.random.rand(*a.shape) > rate).astype(float)
        return (a * mask) / (1 - rate), mask

    def initialise_weights(self):
        for i in range(1, len(self.layer_sizes)):
            weight_matrix = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) * np.sqrt(2. / self.layer_sizes[i - 1])
            bias_matrix = np.zeros((1, self.layer_sizes[i]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

    def forward_pass(self, X, training=True):
        z_list = []
        activations_list = [X]
        dropout_masks = []

        for i in range(len(self.weights)):
            z = np.dot(activations_list[-1], self.weights[i]) + self.biases[i]
            z_list.append(z)

            if i == len(self.weights) - 1: # output layer
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
                if training and self.dropout > 0:
                    a, mask = self.apply_dropout(a, self.dropout)
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)                    

            activations_list.append(a)

        return activations_list[-1], activations_list, z_list, dropout_masks

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_true = y_true.reshape(y_pred.shape)  # ensure shapes match
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backpropagation(self, y_true, y_pred, activations_list, z_list, dropout_masks):
        dL_dz = y_pred - y_true  # Gradients of the loss with respect to z at the output layer
        eps = 1e-8
        beta1, beta2 = 0.9, 0.999

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights))):
            z = z_list[i] if i < len(z_list) else None
            a_prev = activations_list[i]
            
            # derivatives for hidden layers
            if i != len(self.weights) - 1:
                if self.activation == 'relu':
                    grad_act = self.relu_derivative(z)
                else:
                    raise NotImplementedError("Only ReLU is supported")
                dL_dz = dL_dz * grad_act  # Propagate the error backward
                
                # apply dropout mas
                if dropout_masks[i] is not None:
                    dL_dz *= dropout_masks[i]

            dL_dw = a_prev.T @ dL_dz  # Gradient w.r.t. weights for hidden layers
            dL_db = np.sum(dL_dz, axis=0, keepdims=True)  # Sum over batch axis for bias gradients

            # Adam update
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * dL_dw
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (dL_dw ** 2)
            m_hat_w = self.m_weights[i] / (1 - beta1 ** self.t)
            v_hat_w = self.v_weights[i] / (1 - beta2 ** self.t)
            self.weights[i] -= self.learn_rate * m_hat_w / (np.sqrt(v_hat_w) + eps)              
            
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * dL_db
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (dL_db ** 2)
            m_hat_b = self.m_biases[i] / (1 - beta1 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - beta2 ** self.t)
            self.biases[i] -= self.learn_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)  
            
            # propagate to previous layer:
            if i > 0:
                dL_dz = dL_dz @ self.weights[i].T
                
        self.t += 1
                      
    def train_on_batch(self, X, y):
        y_pred, activations_list, z_list, dropout_masks = self.forward_pass(X, training=True)
        self.backpropagation(y, y_pred, activations_list, z_list, dropout_masks)
        loss = self.compute_loss(y, y_pred)
        return loss

    def predict(self, X):
        y_pred, _, _, _ = self.forward_pass(X, training=False)
        return (y_pred > 0.5).astype(int)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

  
sc = StandardScaler()
X = sc.fit_transform(imagedata) # normalise it to 0 and 1 range. Shape [24946, 3072]
y = labels.reshape(-1,1) # shape is now [number of labels, 1]

# split the set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a MLP instance
# mlp = MLP([3072, 512, 128, 1], activation='relu', learn_rate= 0.0005) # default result. loss fluctuates, acc degrading due to overfit
# mlp = MLP([3072, 256, 64, 1], activation='relu', learn_rate= 0.0001) # does well. loss reduces consistently, acc hovers from 0.64 to 0.
# mlp = MLP([3072, 128, 32, 1], activation='relu', learn_rate= 0.0001) # not as good - from 0.645 to 0.6493
# mlp = MLP([3072, 512, 1], activation='relu', learn_rate= 0.0001) # reduce a layer. Result improves. 0.6597 max
mlp = MLP([3072, 256, 1], activation='relu', learn_rate= 0.0001) # BEST result yet. 0.6667 at epoch 19/20. Reducing the neuron in a layer helps
# mlp = MLP([3072, 128, 1], activation='relu', learn_rate= 0.0001) # degrading result. could be overfitting

mlp = MLP3([3072, 512, 128, 1], activation='relu', learn_rate= 0.001, dropout=0.0) # not that much improvement. The limit of MLP on img classification?


# mini-batch gradient descent process
epochs = 20
batch_size = 64

# for each epoch:
for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        loss = mlp.train_on_batch(X_batch, y_batch)
        epoch_loss += loss

    epoch_loss /= (X_train.shape[0] // batch_size)
    acc = mlp.evaluate(X_test, y_test)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Test Accuracy: {acc:.4f}")
    
    
# try some changes
# 1. change batch size: 64 to 128. still degrade over time
# 2. lower learn rate: 0.0005 to 0.0001. definitely improved result
# 3. change model complexity: [3072, 512, 128, 1] to . improved result too
#                                                    [3072, 128, 32, 1]. Most improved with just one hidden layer



