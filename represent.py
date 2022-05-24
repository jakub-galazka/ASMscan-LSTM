import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.manifold import TSNE
from util.model_config import ModelConfig
from util.tokenizer import load_tokenizer
from util.ploter import config_plot, save_plot
from config import REP_COMBS, DATA_PREDICTION_PATH, REP_MODEL, SEP, PREDICTION_COLUMNS, MODEL_PATH, LAYER_BEFORE_CLASSIF_NAME, REPRESENTATION_PATH


model_id = sys.argv[1]
seed = int(datetime.timestamp(datetime.now()))

# Loading model config
config = ModelConfig(model_id)
FOLDS_QUANTITY = config.getParam("folds_quantity")
T = config.getParam("T")

# Loading tokenizer
tokenizer = load_tokenizer()

# Generating representation for each rep combination
for comb in REP_COMBS:
    # Collecting representations for every fold in rep combination 
    rep = []
    for fold in comb:
        # Loading data (from comb model -> significant frags)
        df = pd.read_csv(DATA_PREDICTION_PATH % (model_id, fold.name + "." + (REP_MODEL % "".join(str(i) for i in range(1, FOLDS_QUANTITY + 1)))), sep=SEP)
        frag = df[PREDICTION_COLUMNS[3]]

        # Tokenization
        token_frag = tokenizer.texts_to_sequences(frag)

        # Padding shorter sequences
        data_frag = tf.keras.preprocessing.sequence.pad_sequences(token_frag, T)

        # Collecting fold representations from every cv model
        temp_rep = []
        for fold_no in range(1, FOLDS_QUANTITY + 1):
            # Loading random cv model
            model = tf.keras.models.load_model(MODEL_PATH % (model_id, fold_no))

            # Getting model output before classification layer 
            layer_before_classif_out = model.get_layer(LAYER_BEFORE_CLASSIF_NAME).output
            fun = tf.keras.backend.function(model.input, layer_before_classif_out)
            temp_rep.append(fun(data_frag))
        
        # Concatenating fold representation form every cv model
        temp_rep = np.concatenate(temp_rep, axis=1)
        rep.extend(temp_rep)
        fold.setScope(len(temp_rep))

    # T-distributed Stochastic Neighbor Embedding
    tsne = TSNE(random_state=seed, verbose=2).fit_transform(rep)
    x = tsne[:, 0]
    y = tsne[:, 1]

    # Plotting representation
    config_plot(ylim=False, size_scale=2)
    plt.axis("off")

    i = 0
    rep_name = ""
    for fold in comb:
        scope = fold.scope
        fold_name = fold.name

        plt.scatter(x[i:i+scope], y[i:i+scope], label=fold_name, c=fold.color, s=10)

        rep_name += fold_name + "."
        i += scope

    rep_name = rep_name[:-1]
    save_plot(REPRESENTATION_PATH % (model_id, rep_name))
