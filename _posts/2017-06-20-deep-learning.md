---
layout: post
title: Deep Learning
categories: NeuralNets
---

Deep neural networks (DNNs) have shown significant improvements in several application domains including computer vision and speech recognition [1]. In particular, deep CNNs are one of the most widely used types of deep networks and they have demonstrated state-of-the-art results in object recognition and detection [2,3]. 

While the previous century saw several attempts at creating fast NN-specific hardware and at exploiting standard hardware, the new century brought a deep learning breakthrough in form of cheap, multi-processor graphics cards or GPUs. GPUs excel at the fast matrix and vector multiplications required not only for convincing virtual realities but also for NN training, where they can speed up the learning process by a factor of 50 and more [4]. 

At this point we may ask ourselves: what must a neural network satisfy in order to be called a deep neural network? A straightforward requirement of a DNN follows from its name: it is *deep*. That is, it has multiple, usually more than three, layers of units. This, however, does not fully characterize a deep neural networks. In essence, we often say that a neural network is deep when it has more than three layers and the following two conditions are met [5]:

* The network can be extended by adding layers consisting of multiple units. 

* The parameters of each and every layer are trainable.

From these conditions, it should be understood that there is no absolute number of layers that distinguishes deep NNs from shallow ones. The depth grows by a generic procedure of adding and training one or more layers, until it can properly perform a target task with a given dataset [5].

In classic classification tasks, discriminative features are often designed by hand and then used in a general purpose classifier. However, when dealing with complex tasks such as computer vision or natural language processing, good features that are sufficiently expressive are very difficult to design. A deep model has several hidden layers of computations that are used to automatically discover increasingly more complex features and allow their composition. By learning and combining multiple levels of representations, the number of distinguishable regions in a deep architecture grows almost exponentially with the number of parameters, with the potential to generalize to non-local regions unseen in training [6]. Taking the network depicted in *Figure 1.1.* as an example, the combination of the first four layers work in feature extraction from image and the last fully connected layers in classification. 

![image]({{ site.baseurl }}/public/photos/cnn.jpg){: .center-image }

*Figure 1.1. A simple convolutional neural network. (Source: [www.clarifai.com](www.clarifai.com))*

Nevertheless, DNN are hard to train. We could try to apply stochastic gradient descent by backpropagation algorithm; but there is an intrinsic instability associated to learning by gradient descent in deep networks which tends to result in either the early or the later layers getting stuck during training [7]. In order to avoid that, many factors play an important role for an appropriate train: making good choices of the random weight initialization -a bad initialization can still hamper the learning process-, cost function and activation function [8], applying notably regularization techniques (in order to avoid overfitting) such us dropout and convolutional layers, having a sufficiently large data set and using GPUs. 

<br />
### References

[1] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, and B. Kingsbury, “Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups,” *IEEE Signal Processing Magazine*, vol. 29, no. 6, pp. 82–97, 2012.

[2] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “Xnor-net: Imagenet classification using binary convolutional neural networks,” *CoRR,* vol. abs/1603.05279, 2016.

[3] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large- scale image recognition,” *CoRR*, vol. abs/1409.1556, 2014.

[4] J. Schmidhuber, “Deep learning in neural networks: an overview,” *Neural networks: the official journal of the International Neural Network Society*, vol. 61, pp. 85–117, 2015; 2014.

[5] K. Cho, Foundations and Advances in Deep Learning. Doctoral Thesis, Aalto University, 2014.

[6] G. Parascandolo, Recurrent neural networks for polyphonic sound event detection. Master of Science Thesis, Tampere University of Technology, 2015.

[7] M. A. Nielsen, Neural Networks and Deep Learning. Determination Press, 2015, [neuralnetworksanddeeplearning.com](neuralnetworksanddeeplearning.com).

[8] X. Glorot and Y. Bengio, “Understanding of the difficulty of training deep feedforward neural networks,” *AISTATS 9,* pp. 249–256, 2010.


