# Image classification of cats and dogs - Using K Nearest Neighbour (KNN) Classifier

In this project, I applied the K-Nearest Neighbors (KNN) algorithm to classify images of cats and dogs, and analysed how the model's predictions compared to the actual labels.


What I have learnt:
- Create a pipeline: I developed a complete pipeline using .py scripts instead of Jupyter notebooks. This includes loading image data from folders, preprocessing the images, applying KNN, and evaluating the results — all connected through modular Python files.

- How KNN actually works - While it's easy to apply a model and get predictions, working on a real problem helped me understand how KNN classifies data "under the hood"—by comparing distances to its nearest neighbors in a high-dimensional space.

- Image processing: To make the images comparable, I resized them to a consistent shape. Uniform preprocessing is essential for KNN since the algorithm relies on measuring distances between feature vectors.

- Cross validation: Two metrics are involved in KNN - distance metric (in this case I use euclidian distance) and K value - as in how many neighbours will a vector being assessed? To find the best K value - I ran 5-fold cross validation to find the best K, train KNN with this value, and use it on the test set

## Result:
- The best K value is 19, and the test result accuracy is 0.62 

- This accuracy is significantly better than random guessing (which would be ~0.50), especially considering that KNN is a non-learning model—it makes decisions based solely on proximity to labeled examples.



