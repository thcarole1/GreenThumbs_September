
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras import layers, Sequential, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from params import TRAIN_MDL_DIR

#  ---------------- BASELINE MODEL : NAIVE BAYES --------------------
def get_NB_metric(X,y):
    """ Calculation of baseline with Naive Bayes model"""
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    # Cross-validation
    cv_results = cross_validate(pipeline_naive_bayes,
                                X,
                                y,
                                cv = 10,
                                scoring = ["accuracy"])

    res = cv_results["test_accuracy"].mean()
    res = np.round(res,3)
    return res
# --------------------------------------------------------------------

#  ---------------- RNN MODEL ----------------------------------------
def initialize_model_RNN(vocab_size, embedding_size=50):
    """ Initialize RNN model :
    1. Creates architecture of RNN model
    2. Compiles RNN model
    """
    model = Sequential()
    model.add(layers.Embedding( input_dim = vocab_size+1,
                                output_dim = embedding_size,
                                mask_zero=True,))
    model.add(layers.LSTM(units = 20, activation='tanh'))
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # compile
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(f"✅ RNN model has been initialized")
    return model

def train_model_RNN(X, y, model):
    """Trains RNN model"""
    es = EarlyStopping(patience = 5, restore_best_weights = True)

    history = model.fit(x=X,
                        y=y,
                        batch_size=32,
                        epochs=1000,
                        verbose=0,
                        callbacks=[es],
                        validation_split=0.3)
    print(f"✅ RNN model has been trained")
    return model
# --------------------------------------------------------------------


#  ---------------- MODEL EVALUATION (RNN, CNN) ----------------------
def get_score_evaluation(X_test, y_test, model):
    """ Gets score evaluation"""
    res = model.evaluate(X_test, y_test, verbose=0)[1]
    res = np.round(res,3)
    return res
# --------------------------------------------------------------------

#  ---------------- MODEL PREDICTION (RNN, CNN) ----------------------
def get_prediction(X_test, model):
    """ Gets prediction"""
    prediction = model.predict(X_test)
    return prediction
# --------------------------------------------------------------------

#  ---------------- CNN MODEL ----------------------------------------
def initialize_model_CNN(vocab_size, embedding_size=50):
    """ Initialize RNN model :
    1. Creates architecture of RNN model
    2. Compiles RNN model
    """
    model = Sequential()
    model.add(layers.Embedding( input_dim = vocab_size+1,
                                output_dim = embedding_size,
                                mask_zero=True,))

    model.add(layers.Conv1D(filters = 10,
                            kernel_size = 15,
                            padding='same',
                            activation='relu',))

    model.add(layers.Conv1D(filters = 10,
                        kernel_size = 10,
                        padding='same',
                        activation='relu',))

    model.add(layers.Flatten())
    model.add(layers.Dense(units = 30, activation='relu'))
    model.add(layers.Dropout(rate = 0.15))
    model.add(layers.Dense(units = 1, activation='sigmoid'))

    # Compile
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(f"✅ CNN model has been initialized")
    return model

def train_model_CNN(X, y, model):
    """Trains CNN model"""
    es = EarlyStopping(patience = 5, restore_best_weights = True)

    history = model.fit(x=X,
                        y=y,
                        batch_size=32,
                        epochs=1000,
                        verbose=0,
                        callbacks=[es],
                        validation_split=0.3)
    print(f"✅ CNN model has been trained")
    return model
# --------------------------------------------------------------------


#  ---------------- SAVE MODEL ----------------------------------------
def save_model(model, str):
    """ Saves model"""
    model.save(TRAIN_MDL_DIR + f"{str}_model.keras")

# --------------------------------------------------------------------

#  ---------------- LOAD MODEL ----------------------------------------
def load_model(str):
    """ Loads trained model"""
    return models.load_model(TRAIN_MDL_DIR + f'{str}_model.keras')
# --------------------------------------------------------------------

#  ---------------- SAVE TOKENIZER ----------------------------------------
def save_tokenizer(tokenizer):
    """ Saves tokenizer"""
    # Export Pipeline as pickle file
    with open(TRAIN_MDL_DIR + "tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)
# --------------------------------------------------------------------


#  ---------------- LOAD TOKENIZER ----------------------------------------
def load_tokenizer():
    """ Load tokenizer from pickle filer"""
    loaded_tokenizer = pickle.load(open(TRAIN_MDL_DIR + "tokenizer.pkl","rb"))
    return loaded_tokenizer
# --------------------------------------------------------------------
