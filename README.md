# Stock Prophet

## Overview
Stock Prophet is project for stock price prediction that uses machine learning. Stock prediction is a popular and challenging problem in finance and data science. The goal of this project is to utilize historical stock prices and related data to train a machine learning model that can accurately forecast future stock prices. This can assist investors in making better-informed decisions regarding the purchase or sale of stocks and potentially increase their returns. Various algorithms were implemented and their predictions were compared in order to achieve improved and more precise results. It utilizes [Alpha Vantage API](https://www.alphavantage.co/documentation/).

## [Frontend](https://github.com/ThreeAmigosCoding/StockProphetFrontend)

## Algorithms

### Linear Regression
The implementation of linear regression includes initializing the slope (m) and intercept (c) parameters of the line. For training the model, the least squares formula is used to calculate the optimal slope (m) and intercept (c) for the linear regression model.

### Decision Trees
The decision tree is built using a recursive function. Within this function, the algorithm attempts to find the optimal split point for the dataset and recursively constructs the left and right branches of the tree. This process continues until either the maximum tree depth is reached or the minimum number of samples for splitting is met. Once the tree is built, it is traversed down to the leaf nodes, where decisions are made based on the values contained within. This entire process is repeated for all input data, ultimately resulting in a list of predictions.

### Neural Networks
The neural network is initialized with two weight layers and two biases, along with defined hyperparameters for the learning rate and L2 regularization. The weights are initialized using He initialization, while the biases are set to zero. The activation function used is Leaky ReLU, which helps alleviate the vanishing gradient problem. This function is applied during the forward pass, taking into account the weights and biases for each layer. The weights and biases are updated during the backpropagation method, using the gradient of the loss with respect to the weights, reduced by the learning rate and increased by L2 regularization. When training the network, mini-batch gradient descent is employed, where updates are performed on smaller subsets of the data rather than the entire dataset. This method is efficient and allows for faster convergence.

### Support Vector Machine
The model employed utilizes linear classification. It begins with weight initialization to zero and setting the appropriate hyperparameters. The training set is traversed, and the weights are updated based on the loss. If a sample is correctly classified, an update is performed that reduces the weight. However, if the sample is misclassified, the loss is computed based on the target value and the prediction value, and the weights are updated accordingly. The SVM uses the "hinge" loss, which penalizes misclassified points and points close to the decision boundary. The final prediction result is converted into a class label using the sign function.

## [Poster](https://github.com/ThreeAmigosCoding/StockProphet/blob/dev/Stock%20Prophet.pdf)

## Project Image
![Image](https://github.com/ThreeAmigosCoding/StockProphet/blob/dev/Stock%20Prophet.jpg)

## Authors
- [Miloš Čuturić](https://github.com/cuturic01)
- [Luka Đorđević](https://github.com/lukaDjordjevic01)
- [Marko Janošević](https://github.com/janosevicsm)

## Academic Context
This project was developed for the purposes of the course [Artificial Intelligence](http://www.ftn.uns.ac.rs/21389888/racunarska-inteligencija) and should not be used for making actual stock investments.
### Course Assistant
- [Marko Njegomir](https://github.com/njmarko)
### Course Professor
- Aleksandar Kovačević
