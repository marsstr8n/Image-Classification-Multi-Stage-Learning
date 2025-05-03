import glob 
import os
import numpy as np
from datasets.simpleimageloader import SimpleImageloader
from preprocessing.simplepreprocessor import SimplePreprocessor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# set the args value:
args = {
    "dataset": "datasets/my_images",
    # something to do with logistic regression here
    "jobs": -1
}

# grab the list of imgs to be processed:
print("[INFO] loading images...")

# get all the imagePaths
imagePaths = glob.glob(os.path.join(args['dataset'], '*', '*'))

# sort the paths to be consistent
imagePaths = sorted(imagePaths)

sp = SimplePreprocessor(32,32)
sil = SimpleImageloader(preprocessors=[sp])
(imagedata, labels) = sil.load(imagePaths, verbose=500)
imagedata = imagedata.reshape(imagedata.shape[0], 3072)

# show some info on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
    imagedata.nbytes / (1024 * 1024.0)
))

le = LabelEncoder()
labels = le.fit_transform(labels)
sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(imagedata, labels, stratify=labels, test_size=0.20, random_state=23)

# use standard scaler - all values in range(0,1)
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

clf = LogisticRegression(solver='liblinear', max_iter=20000)


# apply gridsearchCV to find the best params - use l1 + liblinear for now
param_grid = [
    # liblinear supports only l1 and l2 penalties
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1],
        'max_iter': [1000]
    },
    # lbfgs and sag support only l2 penalty
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1],
        'max_iter': [2000]
    },
    # C = 1 not converge for 'sag'
    {
        'solver': ['sag'],
        'penalty': ['l2'],
        'C': [0.01, 0.1],
        'max_iter': [2000]
    },

    # saga with l1 or l2 (no l1_ratio needed). Note - C = [0.01, 1] long time to converge for 'saga' too
    {
        'solver': ['saga'],
        'penalty': ['l1', 'l2'],
        'C': [0.01],
        'max_iter': [1000]
    },
    # saga with elasticnet (l1_ratio required)
    {
        'solver': ['saga'],
        'penalty': ['elasticnet'],
        'C': [0.01],
        'l1_ratio': [0.3, 0.5, 0.7],
        'max_iter': [1000]
    }
]


# set up GridSearchCV
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2
)
print("[INFO] cross validation...")

grid_search.fit(X_train_scaled, y_train)
best_result = grid_search.best_params_, grid_search.best_score_
print("[INFO] the best params to be used", best_result)

best_params = grid_search.best_params_
best_params['random_state'] = 0

clf = LogisticRegression(**best_params)
# clf = LogisticRegression(max_iter=1000, random_state=0)

print("[INFO] training...")


clf.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, clf.predict(X_test_scaled)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
print(classification_report(y_test, clf.predict(X_test_scaled), target_names=le.classes_))
