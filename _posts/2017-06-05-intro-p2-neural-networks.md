---
layout: post
title: Introduction II to Neural Networks
categories: NeuralNets
---

Previous: Introduction I to Neural Networks [here](/blog/neuralnets/2017/06/03/intro-neural-networks/)

## 1. Multilayer Perceptron

Multilayer perceptrons (MLP) constitute one of the simplest type of feedforward NNs (FNNs) and the most popular network for classification and regression [1]. An MLP consists of a set of source nodes forming the input layer, one or more hidden layers of computation nodes, and an output layer. *Figure 1.1.* below depicts the architecture of an MLP with a single hidden layer. 

![image]({{ site.baseurl }}/public/photos/mlp.jpg){: .center-image }

*Figure 1.1. Signal-flow graph of an MLP with one hidden layer. Output layer computes a linear operation.*

For an input vector **x**, each neuron computes a single output by forming a linear combination according to its input weights and then, possibly applying a nonlinear activation function. The computation performed by an MLP with a single hidden layer with a linear output can be written mathematically as:

$$
\begin{equation}
\label{eq:mlp}\tag{1.1}
{\bf \widehat{y}} = {\bf W^{hy}}\cdot\Phi({\bf W^{xh}}{\bf x}+{\bf b}^{h})+{\bf b}^{y},
\end{equation}
$$

where, in vector notation, **W$$^{**}$$** denotes the weight matrices connecting two layers, i.e., **W$$^{xh}$$** are the weights from input to hidden layer and **W$$^{hy}$$** from hidden to output layer, **b$$^{*}$$** are the bias vectors, and the function **$$\Phi(\cdot)$$** is an element-wise non-linearity.

The power of an MLP network with only one hidden layer is surprisingly large. As Hornik et al. and Funahashi showed in 1989 [2,3], such networks, like the one in Equation \eqref{eq:mlp}, are capable of approximating any continuous function $$ \mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$$ to any given accuracy, provided that sufficiently many hidden units are available.

For an input **x** a prediction **$$\widehat{y}$$** is computed at the output layer, and compared to the original target **y** using a *cost function* **$$E(W,b;x,y)$$**, or just *E* for simplicity. The network is trained to minimize *E* for all input samples **x** in the training set, formally:

$$
\begin{equation}
\label{eq:cost}\tag{1.2}
E(W,b)=\frac{1}{N}\sum_{n=1}^{N}{E(W,b;x_{n},y_{n})}
\end{equation}
$$

where N is the number of training samples. Since the *cost function* (also known as *loss* or *objective function*) is a measure of how well our network did to achieve its goal in every epoch, it is a single value. **Mean squared error (MSE)** and *cross entropy* [^fn-sample_footnote] (*H(p,q)* with *p* and *q* two probability distributions) are among the most common cost function to train NNs for classification tasks:

[^fn-sample_footnote]: In information theory, the entropy of a random variable is a measure of the variability associated with it. Shannon defined the entropy of a discrete random variable $X$ as: $$H(X)=-\sum_{x}\mathbb{P}(X=x)\log\mathbb{P}(X=x)$$. From this definition we can deduce straightforward the entropy between two variables (*cross entropy*).

$$
\begin{equation}
\label{eq:mse}\tag{1.3}
E_{MSE}=\frac{1}{N}\sum_{n=1}^{N}{\Vert y_{n}-\widehat{y_{n}}\Vert^{2}}
\end{equation}
$$

$$
\begin{equation}
\label{eq:cross_entropy}\tag{1.4}
E_{CE}=\frac{1}{N}\sum_{n=1}^{N}{H(p_{n},q_{n})} = -\frac{1}{N}\sum_{n=1}^{N}{y_{n}\log\widehat{y_{n}} + (1-y_{n})\log(1-\widehat{y_{n}})} .
\end{equation}
$$

Furthermore, categorical cross entropy is a more granular way to compute error in multiclass classification tasks than simply accuracy or classification error. Let us consider the following example to endorse this statement. Suppose we have two neural networks working on the same problem whose outputs are the probability of belonging to each class, shown in *Table 1.1.* 

*Table 1.1. Example of two networks' output for the same classification problem with three training samples and three different classes. Networks output the probability of belonging to each class; the class with the highest probability is chosen as the solution and compared to the target to decide whether it is correct or not.*

![image]({{ site.baseurl }}/public/photos/tables.jpg){: .center-image }

We choose the class with the highest probability as the solution and then compare it with the known right answer (targets); since both networks classified two items correctly, both present a classification error of *1/3=0.33* and thus, same accuracy. However, while the first network barely classify the first two training items (similar probabilities among all of them), the second network distinctly gets them correct. Should we consider now the average cross entropy error for every network, 

$$
\begin{equation}
  \left\{ \begin{array}{ccc} 
  & & E_{CE}^{1}= -(\log(0.4)+\log(0.4)+\log(0.1))/3 = 1.38, \\
  & &  E_{CE}^{2}= -(\log(0.7)+\log(0.7)+\log(0.3))/3 = 0.64,          
  \end{array}\right.
  \tag{2.1} \label{eq:one}
\end{equation}
$$

we can notice that the second network has a lower value which indicates it actually performed better. The $$log()$$ in cross entropy takes into account the closeness of a prediction.


## 2. Backpropagation

NNs are constructed as differentiable operators and they can be trained to minimize the differentiable cost function using *gradient descent* based methods. An efficient algorithm widely used to compute the gradients for all the weights in the network is the *backpropagation* algorithm, an implementation of the chain rule for partial derivatives along the network. The **backpropagation** algorithm is the most popular learning rule for performing supervised learning tasks [4] and it was proposed for the MLP model in 1986 by Rumelhart, Hinton, and Williams [5]. Later on, the *backpropagation* algorithm was discovered to have already been invented in 1974 by Werbos [6].

Due to *backpropagation*, MLP can be extended to many hidden layers. In order to understand how the algorithm works, we will use the following notation: $$\Phi'$$ is the first derivative of the activation function $$\Phi$$; $$w_{ji}^{l}$$ is the weight connecting the $$i^{th}$$ neuron in the layer $$l-1$$ to the $$j^{th}$$ neuron in the layer $$l$$; $$z_{j}^{l}$$ is the weighted input to the $$j^{th}$$ neuron in layer $$l$$, expressly:

$$
\begin{equation}
\label{eq:zjl}\tag{1.6}
z_{j}^{l}=\sum_{i}{ w_{ji}^{l} \Phi(z_{i}^{l-1}) + b_{j}^{l} } =\sum_{i}{ w_{ji}^{l} h_{i}^{l-1} + b_{j}^{l} },
\end{equation}
$$

where $$h_{i}^{l-1}$$ is the activation of the $$i^{th}$$ neuron in the layer $$l-1$$.
The cost function can be minimized by applying the gradient descent procedure. It requires to compute the derivative of the cost function with respect to each of the weights and bias terms in the network., i.e., $$\frac{\partial E}{\partial w_{ji}^{l}}$$ and $$\frac{\partial E}{\partial b_{j}^{l}}$$. Once these gradients have been computed, the corresponding parameters in the network can be updated by taking a small step towards the negative direction of the gradient. Should we use **stochastic gradient descent** (SGD), 

$$
\begin{equation}
\label{eq:sgd}\tag{1.7}
w\equiv w-\eta \nabla E(w),
\end{equation}
$$

the weights are updated via the following:

$$
\begin{equation}
\label{eq:update}\tag{1.8}
\Delta w_{i}(\tau + 1)=-\eta \nabla E(w_{i})= -\eta\frac{\partial E}{\partial w_{i}},
\end{equation}
$$

where $$\tau$$ is the index of training iterations (epochs); $$\eta$$ is the *learning rate* and it can be either a fixed positive number or it may gradually decrease during the epochs of the training phase. The same update rule applies to the bias terms, with $$b$$ in place of $$w$$.

Backpropagation is a technique that efficiently computes the gradients for all the parameters of the network. Unfortunately, computing $$\frac{\partial E}{\partial w_{ji}^{l}}$$ and $$\frac{\partial E}{\partial b_{j}^{l}}$$ is not so trivial. For MLP, the relationship between the error term and any weight anywhere in the network needs to be calculated. This involves propagating the error term at the output nodes backwards through the network, one layer at a time. First, for each neuron *j* in the output layer *L* an error term $$\delta_{j}^{l}$$ is computed:

$$
\begin{equation}
\label{eq:err}\tag{1.9}
\delta_{j}^{L} \equiv \frac{\partial E}{\partial z_{j}^{L}} = \frac{\partial E}{\partial h_{j}^{L}} \frac{\partial h_{j}^{L}}{\partial z_{j}^{L}}
\end{equation}
$$

We can then compute the backpropagated errors $$\delta_{j}^{l}$$ at the $$l^{th}$$ layer in terms of the backpropagated error $$\delta_{j}^{l+1}$$ in the next layer applying the chain rule:

$$
\begin{equation}
\label{eq:err_1}\tag{1.10}
\delta_{j}^{l} \equiv \frac{\partial E}{\partial z_{j}^{l}} = \sum_{i}\frac{\partial E}{\partial z_{i}^{l+1}} \frac{\partial z_{i}^{l+1}}{\partial z_{j}^{l}}.
\end{equation}
$$

The first factor of Equation \eqref{eq:err_1} can be rewritten directly from definition in \eqref{eq:err} as

$$
\begin{equation}
\label{eq:err_1_1}\tag{1.11}
\frac{\partial E}{\partial z_{i}^{l+1}} \equiv \delta_{i}^{l+1},
\end{equation}
$$

the second factor in Equation \eqref{eq:err_1} can be derived using Equation \eqref{eq:zjl}

$$
\begin{equation}
\label{eq:err_1_2}\tag{1.12}
\frac{\partial z_{i}^{l+1}}{\partial z_{j}^{l}} = \frac{\partial}{\partial z_{j}^{l}}\sum_{i}{w_{ij}^{l+1} h_{j}^{l} + b_{i}^{l+1}} =  \sum_{i}{w_{ij}^{l+1} \Phi'(z_{j}^{l})},
\end{equation}
$$

hence, we can simplify Equation \eqref{eq:err_1}

$$
\begin{equation}
\tag{1.13}
\frac{\partial E}{\partial z_{j}^{l}} = \sum_{i}{\delta_{i}^{l+1} w_{ij}^{l+1} \Phi'(z_{j}^{l})}.
\end{equation}
$$

Finally, the gradients can be expressed in terms of the error $$\delta_{j}^{l}$$

$$
\begin{equation}
\label{eq:gradient_w}\tag{1.14}
\frac{\partial E}{\partial w_{ji}^{l}} = \frac{\partial E}{\partial z_{j}^{l}} \frac{\partial z_{j}^{l}}{\partial w_{ji}^{l}} = h_{i}^{l-1} \delta_{j}^{l}
\end{equation}
$$

$$
\begin{equation}
\label{eq:gradient_b}\tag{1.15}
\frac{\partial E}{\partial b_{j}^{l}} = \frac{\partial E}{\partial z_{j}^{l}} \frac{\partial z_{j}^{l}}{\partial b_{j}^{l}} = \delta_{j}^{l}
\end{equation}
$$

Note that all weights and bias must be initialized to give the algorithm a place to start from. The values are typically drawn randomly and independently from uniform or Gaussian distributions.

The SGD, defined in Equation \eqref{eq:sgd}, is convergent in the mean if $$0<\eta<\frac{2}{\lambda_{max}}$$, where $$\lambda_{max}$$ is the largest eigenvalue of the autocorrelation of the input vector *X*. When $$\lambda$$ is too small, the possibility of getting stuck at a local minimum of the error function is increased. In contrast, the possibility of falling into oscillatory traps is high when $$\lambda$$ is too large. This fact added to the slow convergence of the algorithm lead to several variations to improve performance and convergence speed. 

Following with SGD as the *cost function*, it can also be used in a smarter way to speed up the learning. The idea is to estimate the gradient $$\nabla E(w)$$ by computing $$\nabla E_{x}(w)$$ for a small sample of randomly chosen training inputs, called *batch*, whose size is *m* so that $$m<n$$, with *n* the size of the complete input dataset. By averaging over this sample, provided that the batch size *m* is large enough, it quickly gets a good estimate of the true gradient:[^eq-footnote]

$$
\begin{equation}
\label{eq:batch}
\frac{\sum_{j=1}^{m}{\nabla E_{x_{j}}(w)}}{m} \approx \frac{\sum_{i=1}^{n}{\nabla E_{x_{i}}(w)}}{n} = \nabla E(w).
\end{equation}
$$

[^eq-footnote]: Conventions vary about scaling of the cost function and batch updates. We can omit $$\frac{1}{n}$$, summing over the costs of individual training examples instead of averaging. This is particularly useful when the total number of training examples isn't known in advance.

**Adam** [7] is a recent alternative to SGD. It is a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. Adam was designed to combine the advantages of two other popular techniques: **AdaGrad** [8], which works well with sparse gradients, and **RMSProp** [9], which works well in on-line and non-stationary settings. 

In this chapter we have presented the MLP network, which is the baseline model for FNN. In the upcoming post another type of FNN, Convolutional Neural Networks (CNNs), are described in detail. However, before offering an insight into CNNs, we will briefly present Recurrent Neural Network (RNN). 


### References

[1] L. Grippo, A. Manno, and M. Sciandrone, “Decomposition techniques for multi- layer perceptron training”, *IEEE Transactions on Neural Networks and Learning Systems*, vol. 27, no. 11, pp. 2146–2159, 2016.

[2] K. Hornik, M. Stinchcombe, and H. White, “Multilayer feedforward networks are universal approximators”, *Neural Networks*, vol.2, no.5, pp. 359–366, 1989.

[3] K. Funahashi, “On the approximate realization of continuous mappings by neural networks”, *Neural Networks*, vol.2, no.3, pp. 183–192, 1989.

[4] K.-L. Du and M. N. S. Swamy, "Neural Networks and Statistical Learning", *2014th ed. London: Springer London*, 2014; 2013.

[5] D. E. Rumelhart and J. L. McClelland, *Parallel distributed processing: Vol. 1, Foundations / explorations in the microstructure of cognition* Cambridge, MA: MIT Press, 1987.

[6] P. J. Werbos, *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences.* Doctoral Thesis, Harvard University, 1974.

[7] D. Kingma and J. Ba, “Adam: A method for stochastic optimization” *CoRR*, vol. abs/1412.6980, 2014.

[8] J. Duchi, E. Hazan, and Y. Singer, “Adaptive subgradient methods for on- line learning and stochastic optimization”, *The Journal of Machine Learning Research*, vol. 12, pp. 2121–2159, 2011.

[9] T. Tieleman and G. Hinton, “Lecture 6e: Divide the gradient by a running average of its recent magnitude”, *COURSERA: Neural Networks for Machine Learning*, 2012.

