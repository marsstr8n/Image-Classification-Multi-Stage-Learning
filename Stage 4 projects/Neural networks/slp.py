import numpy as np
from datasets.simpleimageloader import *
from preprocessing.simplepreprocessor import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os
import glob

# set the args value:
args = {
    "dataset": "datasets/my_images",
    # something to do with logistic regression here
    "jobs": -1
}

# load the preprocessed_data.npz
loaded = np.load("preprocessed_data.npz", allow_pickle=True)
imagedata = loaded['data']
labels = loaded['labels'] # 0, 1
class_names = loaded['class_names'] # 'Cat', 'Dog'

print(imagedata[0].shape)


sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(imagedata, labels, stratify=labels, test_size=0.20, random_state=23)

# use standard scaler - all values in range(0,1)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# write the class for the Single Layer Perceptron
# input: list of vectors - taken from my preprocess pipeline
# output: the labeling of either xi is a dog or a cat - to be compared with actual label (y)

class Perceptron:
    def __init__(self, learn_rate=0.01, n_iter=500):
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.errors_ = [] # number of errors for each iteration
        
    def weighted_sum(self, X):
        """Represent the formula ΣXi.wi + b
        """
        return np.dot(X, self.w_) + self.b_ # note: they are initialised in fit
    
    def predict(self, X):
        predictions = []
        for xi in X:
            if self.weighted_sum(xi) > 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
            
    # training method:
    def fit(self, X, y):
        """take in a list containing image vectors, and its corresponding values, also in list form"""
        self.w_ = np.zeros(len(X[1])) # initialise weights based on one dimension of the image vector: 32x32 in this case
        self.b_ = 0
        
        for i in range(self.n_iter):
            error = 0
            for xi, yi in zip(X,y): # use zip so each xi corresponds to its yi
                # calculate predicted y
                pred_y = 1 if self.weighted_sum(xi) > 0 else 0
                
                # calculate the update value
                update = self.learn_rate * (yi - pred_y)
                
                # update weights
                self.w_ = self.w_ + update * xi
                
                # update bias
                self.b_ = self.b_ + update
                
                # have an error update here, every time the iteration makes an error: pred_y != y
                if yi - pred_y != 0:
                    error += 1
            self.errors_.append(error)     
            
            if i % 50 == 0:
                print(f"[INFO] Iteration {i}/{self.n_iter} — Misclassified: {error}")


slp = Perceptron(learn_rate=0.05, n_iter=500)        
slp.fit(X=X_train_scaled, y=y_train)
acc = accuracy_score(y_test, slp.predict(X_test_scaled)) * 100
print(f"Single Layer Perceptron model accuracy: {acc:.2f}%")
print(classification_report(y_test, slp.predict(X_test_scaled), target_names=class_names))

# despite the params learning_rate and max_iter change, the accuracy does not differ much: still at 0.55 (best result), worse than KNN and Logistic Regression


