---
layout: post
title: Understanding LDA for topic modeling 
category: Work
---

Topic modeling provides methods for automatically organizing, searching or even understanding a vast amount of electronic data archives. Latent Dirichlet Allocation (LDA) is one simple method of topic modeling. Although it can be applied to data from different domains such as images, I will focus here on text collections (corpora). But, **what is exactly LDA?** The intuition behind LDA is that documents exhibit multiple topics. Thus, LDA is a way of automatically uncovering the hidden thematic structure in a document collection, which allows us to organize the collection according to the discovered themes (or topics). In fact, LDA is a generative probabilistic model of a corpus: each document is considered to have a set of various topics that are assigned to it via LDA. 


### 1. Key concepts


#### 1.1 Topics

* Each topic is a distribution over terms in our fixed vocabulary. 
* Different topics are composed by different words with different probability.
* Every topic has a probability for every word in the vocabulary, although it might be really low. However, the same word can have high probability in two topics: think of word 'neuron' in a topic related to biology and topic related to machine learning. 
* How many topics there are in a corpora? This variable is defined as $$K$$ and it is a parameter we must set prior training the method (usually between 50 and 150).
* How many topics compose a document? We will get here soon, but this is basically what LDA defines. 


#### 1.2 Generative process for each document

1. Choose a distribution over, let's say, $$K=50$$ topics. This distribution has 50 possible values, which we draw from Dirichlet distribution. 
2. For each word, we choose a topic and look up the distribution over terms associated with that topic and the position of the word (categorical distribution). We iterate repeatedly over words in the document, so the distribution over terms adjust better to the actual content of the document and topics are assigned. Note that here the order of words in the document doesn't matter. 
3. For the next document, we repeat the same process. 


#### 1.3 Graphical model

* Nodes are random variables (observed variables are shaded, hidden variable are uncolored).
* Arrows denote possible dependence between variables.
* Boxes, called plates, denote replicated variables.

![image]({{ site.baseurl }}/public/photos/ldamodel.jpg){: .center-image }

**Notation and observed variables:**

* $$N$$ : vocabulary size
* $$D$$ : number of documents
* $$K$$ : number of topics
* $$W_{d,n}$$ : word $$n$$ in document $$d$$ - all we observe is a bunch of words organized by documents

**Hidden variables** 

Goal: from a collection of documents, infer:

* $$Z_{d,n}$$ : per-word topic assignment
* $$\theta_{d}$$ : per-document topic proportion - one for every document and it has dimension $$K$$ because there are $$K$$ topics
* $$\beta_{K}$$ : per-corpus topic distribution, i.e., the topics themselves - each $$\beta$$ is a distribution over terms and we have $$K$$ of them

* $$\beta$$ comes from a Dirichlet distribution. 

* $$\alpha$$ and $$\eta$$ are Dirichlet parameters: $$\alpha$$ controls the mean shape and sparsity of topics within a document, i.e., $$\theta_{d}$$. With a high value of $$\eta$$, a topic is likely to contain a mixture of most of the words and no a specific set of few words. 


### 2. Dirichlet distribution

Directly from [Wikipedia](https://en.wikipedia.org/wiki/Dirichlet_distribution): *"In probability and statistics, the Dirichlet distribution (after Peter Gustav Lejeune Dirichlet), often denoted $$Dir(\alpha)$$, is a family of continuous multivariate probability distributions parameterized by a vector $$\alpha$$ of positive reals. [...] Dirichlet distributions are very often used as prior distributions in Bayesian statistics."*

So, let's say $$\theta$$ comes from a **3-dimensional Dirichlet**, i.e. $$Dir(\alpha)$$ where $$\alpha$$ is a vector with three dimensions, which represents a distribution over K = 3 elements. We can depict it as in *Figure 2.1*, where the three elements would be each vertex of the triangle. For example, every point between *a* and *b* represents *c* having a probability zero. 

![image]({{ site.baseurl }}/public/photos/dirichlet.png){: .center-image }

*Figure 2.1. Probability density functions of a few Dirichlet distributions (from Wikipedia).*

One very special distribution is if all $$\alpha$$ components are equal to 1, $$Dir(1,1,1)$$ since it is the **uniform distribution** in this space (aprox to **first image** in *Figure 2.1.*, which corresponds to $$Dir(1.3,1.3,1.3)$$). For $$Dir(1/3,1/3,1/3)$$ we are right in the middle of the triangle, but what happens if $$Dir(3,3,3)$$? In this case that $$\alpha_{i}>1$$, it places a hump in the space and if all $$\alpha_{i}$$ have the same value, the hump is centered in the middle (**second image** in *Figure 2.1.*). However, as we can see in *Equation \eqref{eq:mean}* the expectation of the distribution remains $$1/3$$ and it determines the location of the hump, that's why it is also cetered.

$$
\begin{equation}
	\label{eq:mean}\tag{2.1}
	E[\theta_{i}|\alpha] = \frac{\alpha_{i}}{\sum{\alpha_{i}}} = \frac{4}{12} = \frac{1}{3}
\end{equation}
$$

The sum of the $$\alpha$$ components determines the shape of the Dirichlet: the greater the sum of the alphas, the more picky (narrow) the Dirichlet becomes around the mean, i.e. at the point of its expectation. Compare second ($$Dir(3,3,3)$$) and **third image** ($$Dir(7,7,7)$$) in *Figure 2.1.* It is sometimes referred to as *scaling* parameter. 

Finally, what happens if all $$\alpha_{i}<1$$? Or equivalently, $$\sum{\alpha_{i}}<K$$? We get *sparsity*. We end up with a shape that places more probabilities in the corner and we have a concave surface, with and inverted hump. So, only a few topics will have positive probability. This makes no sense with low dimensional Dirichlet, but with fifty or hundred dimensions.  


### 3. Mixture Model

A mixture model assigns just 1 topic to each document. It is quite similar to set $$\alpha$$ close to zero. This scenario is good for finding co-occurring words in one document, since these word will be put in the same topic. 

When should we choose a mixture model? It depends on the corpus. If we are working with a large, heterogeneous corpus, definitely we should opt for a mixture model and set $$\alpha$$~0 (e.g. "I am document 1 and I come from topic 42"). However, if our corpus has different but related topics within each document, we should increase $$\alpha$$'s value yet every document is likely to contain a mixture of topics.


### 4. Practice with LDA 

In order to study the effect of different parameters and how LDA behaves, I set up a corpus with 1000 newspaper articles previously pre-processed (lemmatized and stopwords filtered out). Our **goal** here is to compute $$\theta$$ so we know the composition of each document in the collection. Code valid for Python 2.7 and gensim 2.0.0. 

```python
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

'''
Load a process the corpus here.
OUTPUT: processed_docs : a list of articles, where each article is a list of valis words  (excluding stopwords and punctuation)
'''

dictionary = Dictionary(processed_docs)
dictionary.filter_extremes(no_below=1, no_above=0.6, keep_n=None)
corpus = [dictionary.doc2bow(text) for text in processed_docs]
vocabulary_size = len(dictionary)

# Topic modeling through LDA 
k_topics = 50   # It is usually 50, 100 or 150
alpha = 0.5   # Hyperparameter to adjust 
eta = 0.5   # Hyperparameter to adjust 
niter = 200   # Number of iterations of the algorithm
ldamodel = LdaModel(corpus=corpus, num_topics=k_topics, alpha=alpha, eta=eta, id2word=dictionary, iterations=niter)

# Now let's define the topics with a maximum of n_words relevant words per topic.
n_words = 1000
ldatopics = ldamodel.top_topics(corpus, n_words) 
```

Variable *ldatopics* is a list of 50 topics. Each topic is composed by the top 1000 words, ranked from most likely to least. The following step towards computing $$\theta$$ is to extract these words and define a topic-word matrix to see wich words are more relevant in each topic.

```python
topic_word_matrix = ldamodel.expElogbeta   # [k_topics, vocabulary_size]
word2id = dictionary.token2id
id2word = dict((idx,word) for word,idx in word2id.iteritems())

# A topic in ldatopics contains tuples with words and its probability. 
# Extract the words separetely in a new variable top_words
top_words = []
for topic in ldatopics:
    word_in_topic = []
    for wid in range(n_words):
        word_in_topic.append(topic[0][wid][1])
    top_words.append(word_in_topic)

# Save useful information:
d = ldamodel.get_document_topics(corpus)
topic_doc_matrix = np.zeros((k_topics, len(d)), dtype=np.float16)   # topic-document matrix
topic_x_doc = np.zeros((1, len(d)), dtype=np.int8)   # Number of relevant topics per document (depends on alpha)
list_topic_x_doc=[]   # Store the relevant topic id 

for n,doc in enumerate(d):
    aux = np.reshape(doc, (len(doc), 2))
    topics = aux[:, 0]
    list_topic_x_doc.append(topics)
    if len(topics) > 0:
        topic_x_doc[0, n] = len(topics)
        for i in topics:
            topic_doc_matrix[int(i), n] = aux[int(np.nonzero(aux == int(i))[0])][1]

``` 

So, **we can easily see that *topic_doc_matrix* is $$\theta$$ and *ldatopics* is $$Z_{d,n}$$.** In the following figures we can observe the effect of increasing $$\alpha$$ printing a column (a specific topic) from *topic_doc_matrix*. As explained above, with $$\alpha = 0.05$$ we have a sort of *mixture model* and $$\alpha = 4$$ corresponds with an homogeneous distribution over topics. 

![image]({{ site.baseurl }}/public/photos/theta_alpha005.jpg){: .center-image }
![image]({{ site.baseurl }}/public/photos/theta_alpha07.jpg){: .center-image }
![image]({{ site.baseurl }}/public/photos/theta_alpha4.png){: .center-image }

*Figure 4.1. Dirichlet distribution over 50 topics with different $$\alpha$$ values.*

We can also print distribution of words in each topic selecting a row from *topic_word_matrix*. As shown in *Figure 4.2*, aproximately first 100 words define the topic since the rest of words in vocabulary have very low probability.

![image]({{ site.baseurl }}/public/photos/words_first200_alpha05.jpg){: .center-image }

*Figure 4.2. Distribution of the first 200 words in topic 41, with $$\alpha=0.5$$*

#### Visualization

Alright, we've got to know the basis of LDA. In order to go deeper in our topic' models we can count on visualization tools that make this task much nicer. [**pyLDAvis**](https://github.com/bmabey/pyLDAvis) is an easy-to-use Python library for interactive topic model visualization. By default the topics are projected to the 2D plane, as we can see in the following display. *Figure 4.3.* is just an screenshot to show how it looks when we have only 2 different topics (in this case I mixed up forecast news and children's tales). As depicted, the topic selected contains all relevant terms related to weather. 

![image]({{ site.baseurl }}/public/photos/lda_vis.png){: .center-image } 

*Figure 4.3. Topic visualization rendered by pyLDAvis. The topic selected (red) is mainly build up by weather terms since it is present in documents about forecast news. On the other hand, topic 1 has more relevant terms within tales' vocabulary.*


<div class="message">
  And now... How can we use all of this? Read the post <a href="/blog/neuralnets/2017/12/06/summary-lda/">Text summarization with LDA</a>
</div>


### References
[1] [Topic Models](http://www.cs.columbia.edu/~blei/papers/BleiLafferty2009.pdf); David Blei and John D. Lafferty, 2009.

[2] [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf); David Blei, Andre Y. Ng and Michael I. Jordan, 2003.

[3] [Topic Models & LDA Video Lectures by David Blei](http://videolectures.net/mlss09uk_blei_tm/)


