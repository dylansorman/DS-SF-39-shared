
# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Natural Language Processing and Text Classification
DS | Lesson 11

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*

- Define natural language processing
- Demonstrate how to classify text or documents using `scikit-learn`


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 10 min  | [Opening](#opening)  | Decision Trees and Random Forests |
| 30 mins  | [Introduction](#introduction-classification)   | Text Classification  |
| 30 mins  | [Demo](#demo-text-sklearn)  | Text Classification with scikit-learn |
| 50 mins  | [Independent Practice](#ind-practice)  | Text Classification with scikit-learn  |
| 20 mins  | [Conclusion](#conclusion)  |   |

---

Slides are [here](./assets/slides/11-text-classification.pdf)

<a name="opening"></a>
## Review: Decision Trees and Random Forests  (10 mins)
Recall definitions of Decision Trees and Random Forests from previous lesson.

**Check:** What are some important features of decision trees and random forests?

  - Decision trees are weak learners that are easy to overfit
  - Random forests are strong models that made up a collection of decision trees
    - They are non-linear (while logistic regression is linear)
    - They are mostly black-boxes (no coefficients, but we do have a measure of feature importance)
    - They can be used for classification or regression

<a name="introduction-nlp"></a>
## Introduction: Natural Language Processing (30 mins)

### What is Natural Language Processing? (NLP)

Natural language processing is the task of extracting meaning and information from text documents. There are many types of information we might want to extract; these include simple classification tasks, such as deciding what category a piece of text falls into or what tone it has, as well as more complex tasks like translating or summarizing text.

Most AI assistant systems are typically powered by fairly advanced NLP engines. A system like Siri uses voice-to-transcription to record a command and then various NLP algorithms to identify the question asked and possible answers.

For any of these tasks, from classification to translation, a fair amount pre-processing is required to make the text digestible for our algorithms. Typically we need to add some structure to the text (unstructured data) before we can make decisions based on it.



**Check:** How might NLP be applied within your current jobs or final projects? What are some potential use-cases?

***



### Common Problems in NLP

It's important to keep in mind that each of these subtasks are still very difficult because of the complexity of language. Most often we are looking for heuristics to search through large amounts of text data. There is still a lot of active research being done in each of these areas.

In the last few years, there has been less focus on the rule-based systems seen here. Instead, researchers are looking for more flexible approaches. While older techniques first attempt to uncover the rules of the language and then use those rules to understand text, modern approaches do not attempt to _parse_ or understand the structure of a sentence first. Instead, they just rely on which words are being used. We'll see an example of these modern approaches in the next class.

***

<a name="introduction-classification"></a>
## Introduction: Text Classification (30 mins)

Text classification is the task of predicting what category or topic a piece of text is from. For example, we may want to identify whether an article is a sports or a business story.  We may also want to identify whether an article is positive or negative in sentiment.

Typically this is done by using the text as the features input for the model, and - as in previous classifications - using the label (sports or business, positive or negative) as the target output for it to train on.

When we want to include the text as features, we usually create a _binary_ feature for each word. Then each feature boils down to: "does this piece of text contain that word?"

To do this, we must first create a vocabulary, in order to account for all the possible words in our universe. We will do this in a data-driven way, which usually means taking in all of the words that appear in our corpus. We'll then filter them based on occurrence or usefulness.

In doing so, we'll have many encoding or representation questions along the way, such as:

  - Does the order of words matter?
  - Does punctuation matter?
  - Upper or lower case? Should we treat 'Python' as different from 'python'?

The answer to each of these is problem dependent, but all of them will affect our modeling problem.

**Check:** What do you think? Does word order matter? Case? Punctuation? Discuss and explain your reasoning.

Solution:

- [ ] Yes, order of words may matter.
  - This is especially true when trying to predict positive or negative sentiment.
- [ ] Yes, punctuation may matter.
  - In sentiment prediction, saying "amazing!!!" may result in a different tone than "amazing."
- [ ] Yes, letter case may matter.
  - Upper-case words or phrases are usually proper nouns. For instance, "Python" is more likely to refer to a programming language, while "python" may refer to either the programming language or a type of snake.


Note: Classification using words from the text as features is known as **bag-of-words** classification.

**Check:** What is "bag-of-words" classification stand for and when should it be used? What are some benefits to this approach?

***

<a name="demo-text-sklearn"></a>
## Demo / Codealong: Text Processing in scikit-learn (30 mins)

Scikit-learn has many pre-processing utilities that simplify many of the tasks required to convert text into features for a model. These can be found in the `sklearn.preprocesing.text` package.

We will use the StumbleUpon web crawl dataset again and perform a text classification text. Instead of using other features of the webpages, we will use the text content itself to predict whether or not the webpage is 'evergreen'.


#### CountVectorizer

There are built-in utilities to pull out features from text in `scikit-learn` - most importantly, `CountVectorizer`. It converts a collection of text into a matrix of features. Each row will be a sample (an article or piece of text) and each column will be a text feature (usually a count or binary feature per word).

`CountVectorizer` takes a column of text and creates a new dataset - one row per piece of text (i.e. one row per title) and generates a feature for **every** word in the all of the titles.

**REMEMBER**: Using all of the words can be very useful, but we also need to remember to use regularization to avoid overfitting. Otherwise, using rare words may result in the model learning something that isn't generalizable.

For example, if we are attempting to predict sentiment and see an article that has the word "bessst!", we may link this word to positive sentiment. However, very few articles may ever use this word, so it isn't actually very useful for our model.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features = 1000,
                             ngram_range=(1, 2),
                             stop_words='english',
                             binary=True)
```


`CountVectorizer` arguments:

- `ngram_range` - a range of of length of phrases to use
    - `(1,1)` means use all single words
    - `(1,2)` all contiguous pairs of words
    - `(1,3)` all triples etc.
- `stop_words='english'`
    - Stop words are non-content words (e.g. 'to', 'the', 'it', 'at'). They aren't helpful for prediction (most of the time) so this parameter removes them.
- `max_features=1000`
    - Maximum number of words to consider (uses the first N as most frequent)
- `binary=True`
    - To use a dummy column as the entry (1 or 0, as opposed to the count). This is useful if you think a word appearing 10 times is not more important than whether the word appears at all.

Like models or estimators in `scikit-learn`, vectorizers follow a similar interface:  

  - We create a vectorizer object with the parameters of our feature space.
  - We `fit` a vectorizer to learn the vocabulary
  - We `transform` a set of text into that feature space.

There is a distinction between `fit` and `transform` when it comes to splitting datasets into 'training' and 'test' sets. We want to fit (i.e. learn our vocabulary) from our training set. Since choosing features is a part of our model building process, we **should not** look at our test set to do this.

Whenever we want to make predictions, we will need to create a new data point that contains **exactly** the same columns as our model. If feature 234 in our model represents the word 'cheeseburger', then we need to make sure our test or future example also has 'cheeseburger' as feature 234. We can use `transform` to perform this conversion on the test set (and any future dataset) in the same way.

```python

titles = data['title'].fillna('')

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features = 1000,
                             ngram_range=(1, 2),
                             stop_words='english',
                             binary=True)

- Use `fit` to learn the vocabulary of the titles
vectorizer.fit(titles)

- Use `tranform` to generate the sample X word matrix - one column per feature (word or n-grams)
X = vectorizer.transform(titles)
```

#### Random Forest Prediction Model
Build a random forest model to predict the "evergreeness" of a website using the title features

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 20)

- Use `fit` to learn the vocabulary of the titles
vectorizer.fit(titles)

- Use `transform` to generate the sample X word matrix - one column per feature (word or n-grams)
X = vectorizer.transform(titles)
y = data['label']

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(model, X, y, scoring='roc_auc')
print('CV AUC {}, Average AUC {}'.format(scores, scores.mean()))
```

#### Term Frequency - Inverse Document Frequency (TF-IDF)

An alternative representation of the _bag-of-words_ approach from `CountVectorizer` is a TF-IDF representation. TF-IDF stands for **Term Frequency - Inverse Document Frequency**.

As opposed to using the count of words as features, TF-IDF uses the product of two intermediate values, the _Term Frequency_ and the _Inverse Document Frequency_.

The Term Frequency is equivalent to `CountVectorizer` features, or the number of times (i.e. 'count') that a word appears in the document. This is our most basic representation of text.

To define Inverse Document Frequency, first let's define Document Frequency. **Document Frequency** is the % of documents that a particular word appears in. For example, you could assume `the` appears in 100% of documents, while words like `Syria` would have relatively low document frequency.  

**Inverse Document Frequency** is simply `1 / Document Frequency` (although sometimes this is altered to `log(1 / Document Frequency)`).

Looking at our final term:
  Term Frequency * Inverse Document Frequency = Term Frequency / Document Frequency.  

The intuition behind a TF-IDF representation is that words that have high weight are those that either appear frequently in this document or appear rarely in other documents (therefore unique to this document).

This is a good alternative to using a static set of stop-words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
```

`TfidfVectorizer` follows the same `fit` and `fit_transform` interface of `CountVectorizer`.

**Check:** What does TF-IDF stand for? What does this function do and why is it useful?

**Check:** Use `TfidfVectorizer` to create a feature representation of the stumbleupon titles.

***

<a name="ind-practice"></a>
## Independent Practice: Text Classification in scikit-learn (30 mins)

Tie together the text features of the title with one more feature sets from the previous random forest model. Train this model to see if this improves the AUC.
- Use the `body` text instead of the `title` text - is this an improvement?
- Use `TfIdfVectorizer` instead of `CountVectorizer` - is this an improvement?

**Check:** Were you able to prepare a model that uses both quantitative features and text features? Does this model improve the AUC?

***

<a name="conclusion"></a>
## Conclusion (5 mins)

Let's review:

- Natural language processing is the task of pulling meaning and information from text
- After we have structured our text, we can identified features for other tasks, including: classification, summarization, and translation.
- In `scikit-learn` we use vectorizers to create text features for classification, such as `CountVectorizer` or `TfIdfVectorizer`

***

### ADDITIONAL RESOURCES
- [Natural Language Understanding: Foundations and State of the Art](icml.cc/2015/tutorials/icml2015-nlu-tutorial.pdf)
- [Text Mining Online](http://textminingonline.com/)
