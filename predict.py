import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from util.model_config import ModelConfig
from util.tokenizer import load_tokenizer
from util.data_loader import load_tst_fold
from config import TST_FOLDS, PREDICTION_COLUMNS, SEP, DATA_PREDICTION_PATH, MODEL_PATH, MODEL_NAME


model_id = sys.argv[1]

# Loading model config
config = ModelConfig(model_id)
FOLDS_QUANTITY = config.getParam("folds_quantity")
T = config.getParam("T")

# Loading tokenizer
tokenizer = load_tokenizer()

# Predicting for each test fold
for key in TST_FOLDS:
    # Result holders
    y_pred_frag_s = []

    # Loading data 
    x_tst, ids = load_tst_fold(TST_FOLDS[key]["path"])

    # Sequence fragmentation
    x_tst_frag = []
    scopes = []

    for x in x_tst:
        x_len = len(x)     

        if (x_len > T):
            frags_quantity = x_len - T + 1

            for i in range(frags_quantity):
                x_tst_frag.append(x[i:i+T])

            scopes.append(frags_quantity)
        else:
            x_tst_frag.append(x)
            scopes.append(1)

    # Tokenization
    token_tst_frag = tokenizer.texts_to_sequences(x_tst_frag)

    # Padding shorter sequences
    data_tst_frag = tf.keras.preprocessing.sequence.pad_sequences(token_tst_frag, T)


    # Functions for predicting
    #----------------------------------------------------------------------------------------------------
    def frags_to_seqs_prediction(y_pred_frag):
        i = 0
        y_pred = []
        frags = []

        for scope in scopes:
            scope_y_pred_frag = y_pred_frag[i:i+scope]
            max_y_pred_frag_index = np.argmax(scope_y_pred_frag)

            y_pred.append(scope_y_pred_frag[max_y_pred_frag_index])
            frags.append(x_tst_frag[i+max_y_pred_frag_index])

            i += scope

        return np.array(y_pred), np.array(frags)

    def save_pred(path, y_pred, frags):
        data = {
            PREDICTION_COLUMNS[0]: ids,
            PREDICTION_COLUMNS[1]: y_pred,
            PREDICTION_COLUMNS[2]: np.full(len(ids), TST_FOLDS[key]["class"]),
            PREDICTION_COLUMNS[3]: frags
        }

        df = pd.DataFrame(data)
        df.to_csv(path, sep=SEP, index=False)

    def predict(file_name):
        y_predict, max_frags = frags_to_seqs_prediction(y_pred_frag)
        save_pred(DATA_PREDICTION_PATH % (model_id, file_name), y_predict, max_frags)
    #----------------------------------------------------------------------------------------------------


    # Predictions
    for fold_no in range(1, FOLDS_QUANTITY + 1):
        # Loading model
        model = tf.keras.models.load_model(MODEL_PATH % (model_id, fold_no))

        # Fragments prediction
        y_pred_frag = model.predict(data_tst_frag, verbose=2)
        y_pred_frag = y_pred_frag.flatten() # [[1], [1], ..., [1]] -> [1, 1, ..., 1]
        y_pred_frag_s.append(y_pred_frag)

        # Saving prediction for each cv model
        predict(key + "." + MODEL_NAME + str(fold_no))

    # Averaging fragments predictions
    y_pred_frag = np.mean(y_pred_frag_s, axis=0)

    # Saving prediction for comb cv models
    predict(key + "." + MODEL_NAME + "comb" + "".join(str(i) for i in range(1, FOLDS_QUANTITY + 1)))
