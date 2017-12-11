---
layout: post
title: Text summarization with LDA
category: Work
---

There are different approaches to automatic text summarization based on Latent Dirichlet Allocation (LDA) topic model [1,2,3]. The one proposed here, it's the most straight forward method. 

The fundamental concepts of this very popular method is explained in [this previous post](/blog/work/2017/12/04/topics-lda/). As a brief reminder of what LDA is, say it is a probabilistic topic modeling. LDA is a way of automatically uncovering the underlying thematic structure in a document collection, which allows us to organize the collection according to the discovered *latent* topics. It requires to set a predefined number of topics and play around with its hyperparameters $$\alpha$$ and $$\eta$$. 

---

LDA returns a topic-document matrix where each document is assigned at least one topic. Topics are distributions over terms in our fixed vocabulary, so each word has a probability in every topic. Therefore, if the current document A is composed by topics 5 and 22 and the word *'family'* appears on it, we define the weight of *'family'* within document A as the sum of its probability in topic 5 and 22. 

We perform alike with every word. Since our goal is to make a summary, LDA results are used to select the most relevant sentences. In order to do so, we add up the weight of every word from a sentence to get the sentence's score. The final score takes into account the relevance of a topic in the document and it's penalized by the length of the sentence. This goes along with the idea that long sentences might be redundant. 

**For instance**, imagine document A is a mixture of topic 5 with probability 0.80 and topic 22 with probability 0.20. It is composed by two sentence of the same length: sentence X gets a score of 0.65 in topic 5 and 0.35 in topic 22, and sentence Y gets 0.3 in topic 5 and 0.7 in topic 22. Which one should we send to the resulting summary? If we average these probabilities by the topic-document information, it's easy to see that sentence X will get a higher score and, therefore, will be the selected sentence. 

This is done in the following snippet of code, assuming that we already have a trained LDA model as in [this post](/blog/work/2017/12/04/topics-lda/):

```python
# Open the output file and write a header
outputfile = codecs.open("results/news_keywords_lda.txt", "w", "utf-8")
header = "LDA settings: alpha = %.2f, beta = %.2f, #topics = %d, iterations = %d" % (alpha, beta, k_topics, niter)
outputfile.write(header)

# Read stopwords file and define punctuation
with codecs.open(stopwordsFilePath, 'r', 'utf-8') as file:
	stopwords = [line.strip() for line in file]
my_punctuation = '!"#$%&\'()*+,-./:;<=>?@[]^_`{|}~'

# Max number of sentences to include in the summary
S = 2
# Iterate over all the documents from the corpus, stored in documents
for n in range(len(documents)):
	sentences = nltk.sent_tokenize(documents[n])  # set spanish.pickle at __init__
	number_of_topics_in_current_document = topic_x_doc[0,n]
	# Initialize matrix whose values will be the sentence's weight in each topic
    topic_sentence_matrix = np.zeros((number_of_topics_in_current_document, len(sentences)), dtype=float)
    # Initialize matrix whose values will be the final weight of each sentence
    weight_sentence = np.zeros((1,len(sentences)))

	for s,sentence in enumerate(sentences):
    	for k_top in range(number_of_topics_in_current_document):
			topic = int(list_topic_x_doc[n][k_top])
			words_in_sent = tokenize(sentence, deacc=False)
            nwords = 0
            for word in words_in_sent:
                nwords += 1
                if len(word) > 1 and word not in stopwords and word not in my_punctuation and word in word2id:
                    topic_sentence_matrix[k_top, s] += topic_word_matrix[topic, word2id[word]]

			if nwords == 0:
				# Negative weight if a sentence has no valid word
                topic_sentence_matrix[k_top, s] = -1.0 
            else: 
                topic_sentence_matrix[k_top, s] *= topic_doc_matrix[topic, n]/float(nwords)
        # Sum across topics in the document to get the accumulate weight of each word in each sentence        
		weight_sentence[0,s] = np.sum(topic_sentence_matrix[:, s])

    sentence_id = np.zeros((1,S), dtype=np.int8)
    for s in range(S):
        sentence_id[0,s] = np.argmax(weight_sentence)
        # Avoid to pick the same sentence
        weight_sentence[0, np.argmax(weight_sentence)] = -1.0

    sentence_id = sorted(sentence_id[0])
    outputfile.write("\n\n- Title: "+X_title[n])
    for s in sentence_id:
        outputfile.write(sentences[s]+"\n")

outputfile.close()
```

It's worthy to say that results vary depending on LDA's settings. Although LDA is a common technique, the truth is that the algorithm is not easy to tune, and results are hard to evaluate. 

<div class="message">
  NOTE:
  <br />Latent Semantic Analysis (LSA) is another technique widely used in extractive text summarization [4,5]. For further information about other automatic text summarization approaches, see [6,7].
</div>


### References

[1] E. Y. Hidayat, F. Firdausillah, K. Hastuti, I. N. Dewi and Azhari, "Automatic Text Summarization Using Latent Drichlet Allocation (LDA) for Document Clustering", *International Journal of Advances in Intelligent Informatics*, Vol 1, No 3, November 2015, pp. 132-139.

[2] J. Bian, Z. Jiang and Q. Chen, "Research On Multi-document Summarization Based On LDA Topic Model", *Proceedings - 2014 6th International Conference on Intelligent Human-Machine Systems and Cybernetics*, IHMSC 2014.

[3] Y.-L. Chang and J.-T. Chien, "Latent Dirichlet learning for document summarization", *ICASSP 2009*.

[4] J. Steinberger and K. Jezek, "Using Latent Semantic Analysis in Text Summarization and Summary Evaluation", *Proceedings of the 7th International Conference ISIM*, January 2004.

[5] M. G. Ozsoy, F. N. Alpaslan and I. Cicekli, "Text summarization using Latent Semantic Analysis", *J. Information Science 37*, 2011, pp. 405-417.

[6] D. Das and A. F.T. Martins, "A Survey on Automatic Text Summarization", November 21, 2007.

[7] F. Kiyani and O. Tas, "A survey on Automatic Text Summarization", *Pressacademia 5*, 2017, pp. 205-213.