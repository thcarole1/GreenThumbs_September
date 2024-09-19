import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_features(data : pd.DataFrame)-> pd.DataFrame :
    """ Retrieve the required features from data"""
    X = data[['ReviewText']]
    print(f"✅ X has been created")
    return X

def transform_target(s):
    """ Transforms reviews :
            1. All reviews below 3 (included) are label encoded to 0
            2. All reviews above 3 (excluded) are label encoded to 1"""
    if s in range(1,4):
        s = 0
    if s in range(4,6):
        s = 1
    return s

def get_target(data : pd.DataFrame) -> pd.Series :
    """ Retrieve the target from data"""
    y = data['ReviewScore']
    y = y.apply(transform_target)
    print(f"✅ y has been created")
    return y

def preprocessing(sentence):
    # Removing whitespaces
    sentence = sentence.strip()

    # Lowercasing
    sentence = sentence.lower()

    # Removing numbers
    sentence = ''.join(char for char in sentence if not char.isdigit())

    # Removing punctuation
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    # Tokenizing
    tokenized = word_tokenize(sentence)

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
    cleaned_sentence = " ".join(lemmatized)

    return cleaned_sentence

def preprocess_features(X):
    """ Returns a Series with preprocessed text"""
    X_preproc = X.apply(preprocessing)
    return X_preproc

def get_fitted_tokenizer(X):
    """ returns a fitted tokenizer"""
    tk = Tokenizer()
    tk.fit_on_texts(X)
    print(f"✅ Fitted tokenizer created")
    return tk

def get_tokenized(X, tokenizer):
    """ return tokenized data"""
    X_tokens = tokenizer.texts_to_sequences(X)

    return X_tokens

def get_padded(X_token, maxlen = 30):
    """ returns padded data"""
    X_pad = pad_sequences(X_token,
                          dtype=float,
                          padding='post',
                          maxlen= maxlen)
    return X_pad
