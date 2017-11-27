---
layout: post
title: Introduction to Neural Networks
categories: NeuralNets
---


## 1. Introduction

The area of study of NNs was originally inspired by the goal of modeling biological neural systems, but has since diverged and become a matter of engineering and achieving good results in machine learning tasks. 

A NN’s topology can be described as a finite subset of simple processing units (called *nodes* or *neurons*) and a finite set of weighted connections between nodes that scale the strength of the transmitted signal, mimicking synapses in the human brain. The behavior of the network is determined by a set of real-valued, modifiable parameters or *weights* **W={w1, w2, ...}** which are tuned in every event, known as *epoch*, of the training process. Neurons in the network are grouped into *layers*. There is one input layer, a variable number of hidden layers that perform intermediate computations and one output layer.

<br />**Supervised and unsupervised learning** <br /> Neural networks do not follow the conventional approach to programming, where we tell the computer what to do, breaking big problems up into many smaller tasks that the computer can easily perform. By contrast, a neural network learns itself from observational data, figuring out its own solution to a current problem. [[1]](http://www.neuralnetworksanddeeplearning.com)

Typically, the network reads an input **x** and associate one or more labels **y**. If the network predicts a label for new unseen data, we say it performs a *classification* task. When a database has a sufficient amount of pairs (**x**,**y**), we can make a computer learn how to classify new unseen data by training it on the known instances from the database. It is the so-called *supervised learning*, that try to find patterns in data as useful as possible to predict the labels.

Hence, it is desirable the network learns to classify new unseen instances and not only the training set. We want to prevent our model from overfitting, i.e., from memorizing training pairs instead of generalizing patterns to any example. A classic methodology to ensure the model has not overfitted is to test it on unseen data whose labels are known and evaluate the accuracy.

In contrast to supervised learning, unsupervised learning is another type of machine learning technique that learns patterns in data without neither label information nor an specific prediction task.

<br />
## 2. Perceptron

![image]({{ site.baseurl }}/public/photos/perceptron.pdf){: .center-image }

In order to understand how neurons and NNs work, it is worth to introduce first the baseline unit for modern research: the perceptron, depicted in the *Figure* above with several inputs **{x1, x2, ..., xN}** ∈ R. Frank Rosenblatt [[2]]({{ site.baseurl }}/public/assets/Rosenblatt-perceptron-1958.pdf) proposed a simple rule to compute the output: the neuron’s output, 0 or 1, is determined whether the weighted sum is less than or greater than some threshold value. Just like the weights, the threshold is a real number which is a parameter of the neuron. Mathematically, see *Equation \eqref{eq:one}*:

$$
\begin{equation}
  output = \left\{ \begin{array}{cccc} 
  0 & if & \sum_{j}{w_j x_j} & \leq\mbox{threshold} \\
  1 & if & \sum_{j}{w_j x_j} & >\mbox{threshold},           
  \end{array}\right.
  \tag{2.1} \label{eq:one}
\end{equation}
$$

where it is easy to infer that by varying the weights and the threshold we can get different models of decision-making. However, *Equation \eqref{eq:one}* can be simplified making two notational changes. First, both inputs and weights can be seen as vectors **$$[x_1, x_{2}, ..., x_{N}]^T$$** and **w** respectively, which allows us to rewrite the summation as a dot product. The second change is to move the threshold to the other side of the inequality, and replace it by what is known as the perceptron's bias, $b\equiv-threshold$. The bias can be seen as a measure of how easy is to get the perceptron to output a 1 [[1]](http://www.neuralnetworksanddeeplearning.com). Thus, the perceptron rule can be rewritten into *Equation \eqref{eq:dos}*:

$$
\begin{equation}
  \tag{2.2} \label{eq:dos}
  output = \left\{ \begin{array}{cccc} 
  0 & \mbox{if} & \textbf{w}\cdot \textbf{x}+b & \leq 0 \\
  1 & \mbox{if} & \textbf{w}\cdot \textbf{x}+b & >0         
  \end{array}\right.
\end{equation}
$$

We can devise a network of perceptrons that we would like to use to learn how to solve a problem. For instance, the inputs to the network might be the raw audio from a soundtrack. And we want the network to learn weights and biases so that the output from the network correctly classifies the chord that is being played one at a time. We can now devise a learning algorithm which can automatically tune the weights and biases to get our network to behave in the manner we want after several epochs. The learning algorithm gradually adjusts the weights and biases in response to external stimuli, without direct intervention by a programmer.  

The problem is that this is not possible if our network contains perceptrons, since a small change in the weights (or bias) of any single perceptron in the network could cause the output of that perceptron to completely flip, say from 0 to 1. And that flip may then cause the behavior of the rest of the network to entirely change in some very complicated way [[1]](http://www.neuralnetworksanddeeplearning.com). 

It is possible to overcome this problem by introducing new types of neurons with a nonlinear behavior, which lead us to introduce a new concept: *activation functions*. The main purpose of nonlinear activation functions is to enable the use of nonlinear classifiers.


### 2.1. Activation Function

An activation function scales the activation of a neuron into an output signal. Any function could serve as an activation function, however there are few activation functions commonly used in NNs:

* **Sigmoid Function**. This is a smooth approximation of the step function used in perceptrons. It is often used for output neurons in binary classification tasks, since the output is in the range [0,1]. It is sometimes referred to as *logistic function*. Mathematically,

$$
\begin{equation}
\sigma(x)=\frac{1}{1+e^{-x}}.
\tag{2.3} \label{eq:tres}
\end{equation}
$$

* **Rectified Linear Unit (ReLU)**. This function avoids saturation problems and vanishing gradients, two of the major problems that arise in deep networks. It is depicted in red in the *Figure* below, where we can see how ReLU grows unbounded for positive values of x,

$$
\begin{equation}
ReLU(x) = max(0,x).
\tag{2.4} \label{eq:cuatro}
\end{equation}
$$

* **Hyperbolic Tangent (tanh)**. This function is used as an alternative to the sigmoid function. Hyperbolic tangent is vertically scaled to output in the range [-1,1]. Thus, big negative inputs to the *tanh* will map to negative outputs and only zero-valued inputs are mapped to zero outputs. These properties make the network less likely to get stuck during training, which could be possible with *sigmoid function* for strongly negative inputs. Mathematically,

$$
\begin{equation}
tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}.
\tag{2.5} \label{eq:cinco}
\end{equation}
$$


![image]({{ site.baseurl }}/public/photos/graphs.pdf){: .center-image }

*Figure: Visual representation of sigmoid (blue), rectified linear unit (ReLU, red) and hyperbolic tangent (tanh, green) activation functions. It can be seen that sigmoid and tanh are both bounded functions.*



### References

[1] M. A. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015, neuralnetworksanddeeplearning.com

[2] F. Rosenblatt, “The perceptron: A probabilistic model for information storage and organization in the brain,” Psychological Review, vol.65, no.6, pp. 386–408, November 1958. 

