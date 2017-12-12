---
layout: post
title: Keywords extraction with LDA
category: Work
---

## 1. Background 

After we've done a practical overview of Latent Dirichlet Allocation (LDA) in the [previous post](/blog/work/2017/12/04/topics-lda/), we can state that the basic idea is that documents are represented as mixtures over latent topics, where each topic is characterized by a distribution over words in vocabulary. 

Topic model is inherent to extract representative content from documents, although results strongly depend on the value given to hyperparameters in Dirichlet distribution. Serve as a brief reminder, $$\alpha$$ controls the mean shape and sparsity of topics within a document (sparsity of $$\theta_{d}$$). $$\eta$$ refers to words distribution. With a high value of $$\eta$$, a topic is likely to contain a mixture of most of the words and no a specific set of few words. 

Let's expose here how to extract **keywords** from each document according to its composition of topics (or topic). In order to make it easier to understand, here it's a snippet of the output file that we are looking for (spanish articles): 

```
LDA settings: alpha = 0.30, eta = 0.50, #topics = 50, iterations = 200

- Title: Dorada con tomillo y limón al horno 
KEYWORDS: dorada, poder, quedar, horno, tomillo

- Title: Cosas para hacer en las playas de Barcelona en invierno 
KEYWORDS: deporte, surf, año, querer, solo
```


## 2. Code (Python)

Since our database is made of random articles from newspapers and magazines, there are a wide range of different themes. Thus, we will work with a low $$\alpha$$ value like in a *mixture model*, i.e. usually each document has only one topic assigned. After several experiments, best results in terms of coherence are obtained with 50 topics, $$\alpha=0.30$$ and $$\eta=0.50$$. 


#### Load libraries 

```python
import sys, codecs
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import tokenize
```


#### Define and train LDA model (same code than in the previous post)

Assuming we already have our LDA model trained in a large corpus of documents [(here)](/blog/work/2017/12/04/topics-lda/), we count on the following information:

* $$\theta$$: topic-document matrix, named *topic_doc_matrix* in code. 
	* Number of relevant topics per document (depends on alpha), *topic_x_doc* in code. 
	* Topic id within each document, *list_topic_x_doc* in code.
* $$Z_{d,n}$$: per-word topic assignment, named *ldatopics* in code. 



#### Define a function to extract N keywords from a document

Given a predefined number of keywords, the weight of each word within the document and its vocabulary, return the keywords. The method for computing the weight of each word is exposed in the next section. 

So, we just have to go through the word's weights and choose the top-N. 

```python
def get_keywords(N, weights, words_in_file):
    idx2word = dict((idx,word) for word,idx in words_in_file.iteritems())
    keywords = ''
    for k in xrange(N):
        if k < (N-1):
            keywords += idx2word[np.argmax(weights)] + ', '
        else:
            keywords += idx2word[np.argmax(weights)]
        weights[np.argmax(weights)] = -1.0  # Avoid to pick the same word
    return keywords
```


#### Extract the keywords

The main idea here is to compute word's weights. If it's a stopword or a punctuation mark, its value remains zero. Otherwise, we assign the probability mass of that word in that topic. If the document is composed by more than one topic, sum word's probability from each topic. Then, it is weighted by the posterior probability of the topic within the document, $$\theta$$. 

```python
# Open the output file and write a header
outputfile = codecs.open("results/news_keywords_lda.txt", "w", "utf-8")
header = "LDA settings: alpha = %.2f, beta = %.2f, #topics = %d, iterations = %d" % (alpha, beta, k_topics, niter)
outputfile.write(header)

# Read stopwords file and define punctuation
with codecs.open(stopwordsFilePath, 'r', 'utf-8') as file:
	stopwords = [line.strip() for line in file]
my_punctuation = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'

# Iterate over all the documents from the corpus, stored in X_train_raw
for n in range(len(X_train_raw)):
    outputfile.write("\n\n- Title: " + X_title[n])

    tokens_in_file = list(tokenize(X_train_raw[n], deacc=False))
    word2id_in_file = Dictionary([tokens_in_file]).token2id
    # Initialize a matrix whose values will be each word's weight.
    # Shape: [number_of_topics_in_current_document, number_of_words]
    number_of_topics_in_current_document = topic_x_doc[0,n]
    topic_filewords_matrix = np.zeros((number_of_topics_in_current_document, len(word2id_in_file)), dtype=float)

    for k_top in range(number_of_topics_in_current_document):
        topic = int(list_topic_x_doc[n][k_top])
        for word in word2id_in_file:
            if len(word) > 1 and word not in stopwords and word not in my_punctuation and word in word2id_global:
                topic_filewords_matrix[k_top, word2id_in_file[word]] += topic_word_matrix[topic, word2id_global[word]]

        # Weight by the posterior probability of the topic
        topic_filewords_matrix[k_top,:] *= topic_doc_matrix[topic, n]

    # Sum across topics in the document to get the accumulate weight of each word
    weight_words = np.sum(topic_filewords_matrix, axis=0)
    # Compute keywords
    doc_keywords = get_keywords(KEYWORDS, weight_words, word2id_in_file)
    outputfile.write('KEYWORDS: '+doc_keywords)

outputfile.close()
```


## 3. Conclusion

The application of keyword extraction could vary from simply getting automatic tags from a text, to classify documents or even help in making a coherent summary of the text. A possible handicap of this method is that keywords must be written in the document, so if an article is talking about holidays in Bali but it doesn't mention 'Indonesia', it can't be a salected keyword even though it would fit well. A step forward would be to study the behaviour of this trained LDA model with texts outside the corpus.

<div class="message">
  For further information about the application of this algorithm, read the post <a href="/blog/work/2017/12/07/summary-lda/">Text summarization with LDA</a>. See also the implication of keywords in doc2vec [UNDER CONSTRUCTION].
</div>

