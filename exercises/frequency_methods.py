# Using scikit learn
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'This is life. This is my dream. I like it here. My feet are cold.'
]

vectorizer = TfidfVectorizer()
# for word-level n-grams
# vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
# for character-level n-grams
# vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5))

X = vectorizer.fit_transform(corpus)

print(X.shape)
print(X.toarray())

feature_names = vectorizer.get_feature_names_out()
corpus_index = [n for n in corpus]
df = pd.DataFrame(X.T.todense(), index=feature_names, columns=corpus_index)
print(df)



# TODO Idea for exercise: Movie review classifier using tf-idf
# https://www.analyticsvidhya.com/blog/2021/09/creating-a-movie-reviews-classifier-using-tf-idf-in-python/
