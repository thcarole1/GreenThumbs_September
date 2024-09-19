# Basic libraries
import pandas as pd
import numpy as np
from colorama import Fore, Style

from sklearn.model_selection import train_test_split
# from tensorflow.keras import models

from params import DUMMY_DATA_DIR, ORIGINAL_DATA_DIR

# Import from .py files
from ml_logic.data import retrieve_cleaned_data,\
                            save_original_data
from ml_logic.model import get_NB_metric,get_score_evaluation,\
                            initialize_model_RNN, train_model_RNN, \
                            initialize_model_CNN, train_model_CNN, \
                            save_model, load_model, get_prediction,\
                            save_tokenizer,load_tokenizer

from ml_logic.preprocessor import get_features, get_target, \
                                preprocess_features, get_fitted_tokenizer, \
                                get_tokenized, get_padded

def main_program():
        # Retrieve the data and clean it
        reviews_df = retrieve_cleaned_data()

        # Get X and y from data
        X = get_features(reviews_df)
        y = get_target(reviews_df)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        print(f"✅ Train test split Done")

        # Preprocess X_train and X_test
        X_train_preproc = preprocess_features(X_train['ReviewText'])
        print(f"✅ X_train preprocessed")
        X_test_preproc = preprocess_features(X_test['ReviewText'])
        print(f"✅ X_test preprocessed")

        # Baseline calculation
        NB_metric_accuracy = get_NB_metric(X_train_preproc, y_train)
        print(f"✅ Baseline accuracy calculated !")
        print(f"⭐️ Baseline accuracy : {NB_metric_accuracy}")

        # ------- Preprocessing for Neural Networks !!---------
        # Tokenize
        tokenizer = get_fitted_tokenizer(X_train_preproc)
        # save fitted tokenizer
        save_tokenizer(tokenizer)

        X_train_tokens = get_tokenized(X_train_preproc, tokenizer)
        X_test_tokens = get_tokenized(X_test_preproc, tokenizer)
        print(f"✅ Tokenized data created")
        # Vocab size?
        vocab_size = len(tokenizer.word_index)
        print(f"✅ vocab_size has been retrieved")

        # Padding
        X_train_pad = get_padded(X_train_tokens, maxlen = 30)
        X_test_pad = get_padded(X_test_tokens, maxlen = 30)
        print(f"✅ Padded data created")
        # -----------------------------------------------------

        # ------------------ RNN ------------------------------
        # Architecture
        model = initialize_model_RNN(vocab_size, embedding_size=50)
        # Training
        model_trained = train_model_RNN(X_train_pad, y_train, model)
        # Evaluation
        RNN_metric_accuracy =get_score_evaluation(X_test_pad, y_test, model_trained)
        print(f"✅ RNN score has been evaluated")
        print(f"⭐️ RNN accuracy : {RNN_metric_accuracy}")
        # -----------------------------------------------------

        # ------------------ CNN ------------------------------
        # Architecture
        model = initialize_model_CNN(vocab_size, embedding_size=50)
        # Training
        model_trained = train_model_CNN(X_train_pad, y_train, model)
        # Evaluation
        CNN_metric_accuracy =get_score_evaluation(X_test_pad, y_test, model_trained)
        print(f"✅ CNN score has been evaluated")
        print(f"⭐️ CNN accuracy : {CNN_metric_accuracy}")
        # -----------------------------------------------------

       #--------------------- Summary -------------------------
        metrics = {'model' : ['NB (baseline)','RNN','CNN'],
            'accuracy' : [NB_metric_accuracy, RNN_metric_accuracy, CNN_metric_accuracy]}

        metrics = pd.DataFrame(metrics).sort_values(by = 'accuracy', ascending=False)
        print(f"⭐️⭐️ Summary ⭐️⭐️")
        print(metrics)
       # -----------------------------------------------------


def train_model():
        # Retrieve the data and clean it
        reviews_df = retrieve_cleaned_data()

        # Get X and y from data
        X = get_features(reviews_df)
        y = get_target(reviews_df)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        print(f"✅ Train test split Done")

        # Preprocess X_train and X_test
        X_train_preproc = preprocess_features(X_train['ReviewText'])
        print(f"✅ X_train preprocessed")
        X_test_preproc = preprocess_features(X_test['ReviewText'])
        print(f"✅ X_test preprocessed")

                # ------- Preprocessing for Neural Networks !!---------
        # Tokenize
        tokenizer = get_fitted_tokenizer(X_train_preproc)
        X_train_tokens = get_tokenized(X_train_preproc, tokenizer)
        X_test_tokens = get_tokenized(X_test_preproc, tokenizer)
        print(f"✅ Tokenized data created")
        # Vocab size?
        vocab_size = len(tokenizer.word_index)
        print(f"✅ vocab_size has been retrieved")

        # Padding
        X_train_pad = get_padded(X_train_tokens, maxlen = 30)
        X_test_pad = get_padded(X_test_tokens, maxlen = 30)
        print(f"✅ Tokenized data created")
        # -----------------------------------------------------
        # ------------------ RNN Training ------------------------------
        # Architecture
        model = initialize_model_RNN(vocab_size, embedding_size=50)
        # Training
        model_trained = train_model_RNN(X_train_pad, y_train, model)
        print(f"⭐️ RNN  has been trained")

        save_model(model_trained, 'RNN')
        print(f"⭐️ Trained RNN model saved")
        # -----------------------------------------------------


def test_prediction():
    X_test = pd.read_json(DUMMY_DATA_DIR + '20240918_004705_dummy_3_reviews.json')
    X_test_preproc = preprocess_features(X_test['ReviewText'])

    # ------- Preprocessing for Neural Networks !!---------
    # Tokenize
    tokenizer = load_tokenizer()
    print(f"✅ Tokenizer loaded")
    X_test_tokens = get_tokenized(X_test_preproc, tokenizer)
    print(f"✅ Tokenized data created")

    # Padding
    X_test_pad = get_padded(X_test_tokens, maxlen = 30)
    print(f"✅ Padded data created")
    # ----------------------------------------------------

    #  Load RNN model
    model = load_model('RNN')
    print(f"✅ Model has been loaded ! ")

    # Prediction
    prediction = get_prediction(X_test_pad, model)
    prediction = prediction.tolist()
    prediction_final = [_[0] for _ in prediction]
    print(prediction_final)
    print(f"✅ Prediction Done ! ")

def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
        # main_program()
        # train_model()
        save_original_data()
        # test_prediction()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
