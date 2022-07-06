import os
import sys
import json
import random
import numpy as np
import tensorflow as tf

from datetime import datetime
from util.tokenizer import load_tokenizer
from util.preprocesser import random_pad_seqs
from util.data_loader import load_trn_val_fold
from config import TEST_MODE_LABEL, DIRS_TREE, FOLDS_QUANTITY, T, V, D, M, LAST_RETURN_SEQS_LAYER_NAME, LAYER_BEFORE_CLASSIF_NAME, MODEL_NAME, EPOCHS, MODEL_PATH, DATA_HISTORY_PATH, MODEL_ARCHITECTURE_PATH, MODEL_CONFIG_PATH


# Testing mode
test_mode = sys.argv[1] == "-t"
if test_mode:
    seed = 1
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    FOLDS_QUANTITY = seed
    MODEL_NAME = MODEL_NAME + TEST_MODE_LABEL 

# Model ID
model = None
timestamp = str(datetime.timestamp(datetime.now()))
model_id = (timestamp + TEST_MODE_LABEL) if test_mode else timestamp

# Crating dir tree
for dir in DIRS_TREE:
    os.makedirs(dir % model_id)

# Loading tokenizer
tokenizer = load_tokenizer()

for fold_no in range(1, FOLDS_QUANTITY + 1):
    print("____________________ Training Fold No. %d / %d ____________________" % (fold_no, FOLDS_QUANTITY))

    # Loading data
    x_trn, y_trn, x_val, y_val = load_trn_val_fold(fold_no)

    # Random padding shorter sequences
    random_pad_seqs(x_trn)
    random_pad_seqs(x_val)

    # Tokenization
    token_trn = tokenizer.texts_to_sequences(x_trn)
    token_val = tokenizer.texts_to_sequences(x_val)

    # Reshaping to numpy.ndarray
    data_trn = np.array(token_trn)
    data_val = np.array(token_val)

    # Creating model
    i = tf.keras.layers.Input(shape=(T,), name="input")
    x = tf.keras.layers.Embedding(V, D, name="embedding")(i)
    '''
        Bidirectional (https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66)

        return_sequences=True -> (N,T,D) == (samples, time_steps, hidden_state_dim)
        return_sequences=False -> return_sequences[:,-1] (last_time_step)

        Bidirectional wrapper when return_sequences=False -> fun([forward_output, backward_output]) -> r = fun([(1..n), (n..1)]) -> output: r[-1] == fun([forward_n_time_step, backward_1_time_step])
        Bidirectional wrapper when return_sequences=True -> fun([forward_output, reverse(backward_output)]) -> output: r = fun([(1..n), (1..n)]) -> r[-1] == fun([forward_n_time_step, backward_n_time_step])
        
        *fun = {sum, mul, concat (default), ave, None}
    '''
    x = tf.keras.layers.LSTM(M, return_sequences=True, name="bi-lstm-forward")(x)                       # forward_output = (1..n)
    y = tf.keras.layers.LSTM(M, return_sequences=True, go_backwards=True, name="bi-lstm-backward")(x)   # backward_output = (n..1)
    x = tf.keras.layers.Concatenate(axis=2, name=LAST_RETURN_SEQS_LAYER_NAME)([x, y])
    x = tf.keras.layers.Lambda(lambda x: x[:,-1], name="bi-lstm-out")(x)
    x = tf.keras.layers.Dropout(.75, name=LAYER_BEFORE_CLASSIF_NAME)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="dense")(x)
    model = tf.keras.models.Model(i, x, name=MODEL_NAME)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.SensitivityAtSpecificity(.99, name="sens_at_spec_99"),
            tf.keras.metrics.SensitivityAtSpecificity(.999, name="sens_at_spec_999")
        ]
    )

    # Training model
    r = model.fit(
        data_trn,
        y_trn,
        epochs=EPOCHS,
        validation_data=(data_val, y_val),
        verbose=2
    )

    # Saving results
    model.save(MODEL_PATH % (model_id, fold_no))
    np.save(DATA_HISTORY_PATH % (model_id, fold_no), r.history)

# Saving model info
with open(MODEL_ARCHITECTURE_PATH % model_id, "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

with open(MODEL_CONFIG_PATH % model_id, "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "folds_quantity": FOLDS_QUANTITY,
        "T": T
    }, f)
