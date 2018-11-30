---
layout: post
title: Text-to-Pictogram summarization [paper presentation] 
category: Work
---

## Introduction 

Many people suffer from **language disorders** that affect their communicative capabilities. Augmentative and alternative communication devices assist learning process through graphical representation of common words. In our paper [Text-to-Pictogram Summarization for Augmentative and Alternative Communication](http://dx.doi.org/10.26342/2018-61-1) published at [SEPLN](http://www.sepln.org), we present a complete text-to-pictogram system able to simplify complex texts and ease its comprehension with pictograms.

![image]({{ site.baseurl }}/public/photos/header_picto.png){: .center-image } 

<div class="message">
  <b>Augmentative... WHAT?</b>
  <br><br>Augmentative and Alternative Communication (AAC) is a form of expression different from regular spoken language, which 'add-on' or replace speech when there are difficulties in communication. 
  <br>AAC includes simple systems such as pictures, gestures and pointing, as well as more complex techniques involving powerful computer technology. 
</div>


## Motivation 

Malinowski suggests, language is *"the necessary means of communion; it is the one indispensable instrument for creating the ties of the moment without which unified social action is impossible"*. Which makes sense, right? After all, language is what define us as humans. Barriers in communication can cause isolation and loss of personal autonomy to complete some daily activities. 

Our main goal was to explore the use of NLP jointly with the [ARASAAC](http://www.arasaac.org) pictographic language to build simplified sentences from a summarized text. Our visual representation aims to capture the core meaning of the input document (newspaper articles, short stories, tales...)


## System Overview

Complete pipeline of our system:

![image]({{ site.baseurl }}/public/photos/pipeline.pdf){: .center-image } 

#### Key elements:

* **Text-summarization**: extractive method based on LSA following Ozsoy et. [1] al approach to extract salient sentences. 

* **Vocabulary** selection: dictionary of 51358 terms with 32 POS labels. We filter out terms with frequency under 100, made use of lemmas and POS information (Freeling).

* **Word embeddings**: train Word2Vec algortihm over words with its POS tag appended, i.e. cat/NC where NC stands for Common Name. Well trained word embeddings are key to find synonyms (limited vocabulary of pictograms) and resolve polysemy.

* **Topic modeling**: LDA with 50 topics trained over half million newspaper articles. It measures the relevance of words in a document, so it helped us to build proper sentence embeddings used for pictogram selection. 


#### Pictogram selection algorithm

Given a sequence of words S 
```
1: for W in S do
2:    If W is not a stopword then
3:       If W is a proper noun then
4:          Picto = GET public domain picture
5:       Else
6:          X = retrieve list of pictos candidates from ARASAAC for word W
7:          If X.length == 0 then
8:             X = GET synonyms of W measuring cosine distance among word embeddings
9:          Picto = SelectPicto(X, original sentence, lemmatized sentence)
10:         If Picto.length == 0 then
11:             Picto = GET public domain picture

```

#### Polysemy resolution

Take as an example of polysemy resolution the following sentence: **Every year CRANES return to the wetlands**

Here, LDA model assigns 2 topics:
* Topic 1 related with nature.
* Topic 2 related with climate change.

Then, our algortihm looks for the best pictogram for *crane*, which is the lemma of *cranes*, and it is able to distinguish the right pictogram and resolve polysemy. 

![image]({{ site.baseurl }}/public/photos/polysemy.png){: .center-image } 


Steps followed:
```
1: Compute *sentence embedding* as a weighted sum of its L word embeddings (L is the length of the target sentence)
2: Compute *tag embeddings* for each pictogram using the tags as a sentence (shown in pictures below).
3: Measure similarity (cosine distance) in the embedding space and select the pictogram with highest score.
```

Since tags from pictogram 1 are closely related to words from original sentence and match the LDA topics, it is properly selected.  

For further information about formulas or a deeper insight into the entire system, have a look [here](http://dx.doi.org/10.26342/2018-61-1).


### References

[1] Ozsoy, M., F. Alpaslan, and I. Cicekli, "Text summarization using latent semantic analysis", *Journal of Information Science*, volume 37, pages 405-417, 2011. 
