---
layout: post
title: Audio Generation
category: Unizar
---

Algorithmic music generation is a difficult task that has been actively explored in earlier decades. Many common methods for algorithmic music generation consist of constructing carefully engineered musical features and rely on simple generation schemes, such as hidden Markov models (HMMs) [1]. It captures the musical style of the training data as mathematical models. Following these approaches the resulting pieces usually consist of repetitive musical sequences with a lack thematic structure.

With the increase in computational resources and recent researches in neural network architectures, novel music generation may now be practical for large scale corpuses leading to better results. Models look after a pleasant to hear outcome since it is not easy to find an objective evaluation of the performance of the network.

Extremely good results are obtained with **WaveNet** model from the paper [2], which works directly at waveform level and uses a very deep dilated convolutional network to generate samples one at a time sampled at 16 KHz. By increasing the amount of dilation at each depth, they are able to capture larger receptive fields and thus, long range dependencies from the audio. Despite the extensive depth, training the network is relatively easy because they treat the generation as a classification problem. It is reduced to classify the generated audio sample into one of 255 values (8 bits encoding). 

Nonetheless, many recent studies that work with raw audio databases agree on RNN as the preferred architecture [3,4,5] to learn underlying dependencies from music input files. Both works [3] and [5] are based on LSTM networks trained with data in the frequency domain of the audio. This enables a much faster performance because it allows the network to train and predict a group of samples that make up the frequency domain rather than one sample [3]. 

In practice it is a known problem of these models to not scale well at such a high temporal resolution as is found when generating acoustic signals one sample at a time, e.g., 16000 times per second. That is the reason why enlarging the receptive field [2] is crucial to obtain samples that sound musical.

It may perhaps be considered without straying too far afield from our primary focus some speech synthesis techniques, since it is one of the main areas within audio generation. Conventional approaches typically use decision tree-clustered context-dependent HMMs to represent probability densities of speech parameters given texts [6,7]. Speech parameters are generated from the probability densities to maximize their output probabilities, then a speech waveform is reconstructed from the generated parameters. This approach has several advantages over the concatenative speech synthesis approach [8], such as the flexibility in changing speaker identities and emotions and its reasonable effectiveness. However, HMMs are inefficient to model complex context dependencies and its naturalness is still far from that of actual human speech. 

Inspired by the successful application of deep neural networks to automatic speech recognition, an alternative scheme based on deep NNs has increasingly gained importance applied to speech generation, although it is worth to emphasize that NNs have been used in speech synthesis since the 90s [9]. In the statistical parametric speech synthesis (SPSS) field [10], DNN-based speech synthesis already yields better performance than HMM-based speech synthesis, provided we have a large enough database and under the condition of using a similar number of parameters [11]. 

Regarding acoustic speech modeling in speech generation, deep learning can also be applied to overcome the limitations from previous approaches. These deep learning approaches can be classified into three categories according to the modeling steps, as well as the relationship between the input and output features represented in the model [12]: 

1. **Cluster-to-feature mapping using deep generative models.** In this approach, the deep learning techniques are applied to the cluster-to-feature mapping step of acoustic modeling for SPSS, i.e., to describe the distribution of acoustic features at each cluster. The input-to-cluster mapping, which determines the clusters from the input features, still uses conventional approaches such as HMM-based speech synthesis [13].

2. **Input-to-feature mapping using deep joint models.** This approach uses a single deep generative model to achieve the integrated input-to-feature mapping by modeling the joint probability density function (PDF) between the input and output features. In [14], the authors propose an implementation with input features capturing linguistic contexts and output features being acoustic features.

3. **Input-to-feature mapping using deep conditional models.** Similar to the previous approach, this one predicts acoustic features from inputs using an integrated deep generative model [15]. The difference is that this approach models a conditional probability density function of output acoustic features, given input features instead of their joint PDF.

<br />
### References

[1] W. Schulze and B. van der Merwe, “Music generation with markov models,” IEEE Multimedia, vol. 18, no. 3, pp. 78–85, 201

[2] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” CoRR, vol. abs/1609.03499, 2016.

[3] V. Kalingeri and S. Grandhe, “Music generation with deep learning,” CoRR, vol. abs/1612.04928, 2016.

[4] S. Mehri, K. Kumar, I. Gulrajani, R. Kumar, S. Jain, J. Sotelo, A. Courville, and Y. Bengio, “Sample rnn: An unconditional end-to-end neural audio generation model,” CoRR, vol. abs/1612.07837, 2016.

[5] A. Nayebi and M. Vitelli, “Gruv : Algorithmic music generation using recur- rent neural networks,” Course CS224D: Deep Learning for Natural Language Processing (Stanford), 2015.

[6] K. Tokuda, Y. Nankaku, T. Toda, H. Zen, J. Yamagishi, and K. Oura, “Speech synthesis based on hidden markov models,” Proceedings of the IEEE, vol. 101, no. 5, pp. 1234–1252, 2013.

[7] H. Zen, K. Tokuda, and T. Kitamura, “Reformulating the hmm as a trajectory model by imposing explicit relationships between static and dynamic feature vector sequences,” Computer Speech & Language, vol. 21, no. 1, pp. 153–173, 2007.

[8] A. J. Hunt and A. W. Black, “Unit selection in a concatenative speech synthesis system using a large speech database,” vol. 1. IEEE International Conference on Acoustics, Speech, and Signal Processing Conference Proceedings (ICASSP), 1996, pp. 373–376.

[9] O. Karaali, G. Corrigan, and I. Gerson, “Speech synthesis with neural net- works,” World Congress on Neural Networks, pp. 45–50, 1996.

[10] H. Zen, K. Tokuda, and A. W. Black, “Statistical parametric speech synthesis,” Speech Communication, vol. 51, no. 11, pp. 1039–1064, 2009.

[11] H. Ze, A. Senior, and M. Schuster, “Statistical parametric speech synthesis using deep neural networks.” IEEE International Conference on Acoustics, Speech, and Signal Processing Conference Proceedings (ICASSP), 2013, pp. 7962–7966.

[12] Z.-H. Ling, S.-Y. Kang, H. Zen, A. Senior, M. Schuster, X.-J. Qian, H. M. Meng, and L. Deng, “Deep learning for acoustic modeling in parametric speech gen- eration: A systematic review of existing techniques and future trends,” IEEE Signal Processing Magazine, vol. 32, no. 3, pp. 35–52, 2015.

[13] Z.-H. Ling, L. Deng, and D. Yu, “Modeling spectral envelopes using restricted boltzmann machines and deep belief networks for statistical parametric speech synthesis,” IEEE Transactions on Audio, Speech and Language Processing, vol. 21, no. 10, p. 21292139, 2013.

[14] S.-Y. Kang, X.-J. Qian, and H. Meng, “Multi-distribution deep belief network for speech synthesis.” IEEE International Conference on Acoustics, Speech, and Signal Processing Conference Proceedings (ICASSP), 2013, p. 80128016.

[15] H. Zen, A. Senior, and M. Schuster, “Statistical parametric speech synthe- sis using deep neural networks.” IEEE International Conference on Acous- tics, Speech, and Signal Processing Conference Proceedings (ICASSP), 2013, p. 79627966.
