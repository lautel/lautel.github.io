---
layout: post
title: Neural Networks
categories: NeuralNets
---


## 1. Introduction

The area of study of NNs was originally inspired by the goal of modeling biological neural systems, but has since diverged and become a matter of engineering and achieving good results in machine learning tasks. 

A NN’s topology can be described as a finite subset of simple processing units (called *nodes* or *neurons*) and a finite set of weighted connections between nodes that scale the strength of the transmitted signal, mimicking synapses in the human brain. The behavior of the network is determined by a set of real-valued, modifiable parameters or *weights* **W={w1, w2, ...}** which are tuned in every event, known as *epoch*, of the training process. Neurons in the network are grouped into *layers*. There is one input layer, a variable number of hidden layers that perform intermediate computations and one output layer.

<br />**Supervised and unsupervised learning** <br /> Neural networks do not follow the conventional approach to programming, where we tell the computer what to do, breaking big problems up into many smaller tasks that the computer can easily perform. By contrast, a neural network learns itself from observational data, figuring out its own solution to a current problem. [1](neuralnetworksanddeeplearning.com)

Typically, the network reads an input **x** and associate one or more labels **y**. If the network predicts a label for new unseen data, we say it performs a *classification* task. When a database has a sufficient amount of pairs (**x**,**y**), we can make a computer learn how to classify new unseen data by training it on the known instances from the database. It is the so-called *supervised learning*, that try to find patterns in data as useful as possible to predict the labels.

Hence, it is desirable the network learns to classify new unseen instances and not only the training set. We want to prevent our model from overfitting, i.e., from memorizing training pairs instead of generalizing patterns to any example. A classic methodology to ensure the model has not overfitted is to test it on unseen data whose labels are known and evaluate the accuracy.

In contrast to supervised learning, unsupervised learning is another type of machine learning technique that learns patterns in data without neither label information nor an specific prediction task.

## 2. Perceptron

In order to understand how neurons and NNs work, it is worth to introduce first the baseline unit for modern research: the perceptron, depicted in Figure 2.1. 

![placeholder]({{ site.baseurl }}/public/photos/perceptron.pdf "Figure 1")




### References

[1] M. A. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015, neuralnetworksanddeeplearning.com