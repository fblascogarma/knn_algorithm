# Performant Predictive Model using KNN algorithm

## About this project

The goal is to develop a model that can predict the age of an abalone (small sea snails; see image below) based purely on physical measurements (except the number of rings on its shell as this kills them).

First, we will build a basic kNN model, and then we are going to fully tuned it to increase performance of predictions.

You can clone this repo and install the requiered libraries by typing pip install -r requirements.txt. Then, it's ready to go.

An abalone:
![alt text](https://files.realpython.com/media/LivingAbalone.b5330a55f159.jfif)

This is a project that follows a tutorial from [RealPython](https://realpython.com/knn-python/) by Joos Korstanje.

## The k-Nearest Neighbors (kNN) Algorithm in Python

The kNN algorithm is one of the most famous machine learning (ML) algorithms. Generally, it is used for classification. KNN works with small datasets and when you need to classify the data in more than two categories.

It is a supervised ML algorithm; i.e., it predicts a target variable using one or multiple independent variables. Also, it's a nonlinear learning algorithm; i.e., does not use a line to separate the data points.

It can be used for classification and regression (separetly or combined), and it is a little bit atypical as compared to other ML algorithms because the specific formula is computed at the moment of prediction instead of at the moment of fitting.

## Drawbacks

Unlikely to perform well on advanced tasks like computer vision and natural language processing.

## How does the algorithm work?

When a new data point arrives, the kNN algorithm, as the name indicates, will start by finding the nearest neighbors of this new data point. Then it takes the values of those neighbors and uses them as a prediction for the new data point.