import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from cycler import cycler
from util.tsne import tsne_2d
from util.model_config import ModelConfig
from util.tokenizer import load_tokenizer
from util.ploter import config_plot, save_plot
from config import REP_COMBS, DATA_PREDICTION_PATH, SEP, PREDICTION_COLUMNS, MODEL_PATH, LAYER_BEFORE_CLASSIF_NAME, COLORS_CYCLE, MARKER_SIZE, TYPE_SEPARATION_LABEL, REPRESENTATION_PATH, comb_model_name


model_id = sys.argv[1]

# Loading model config
config = ModelConfig(model_id)
MODEL_NAME = config.getParam("model_name")
FOLDS_QUANTITY = config.getParam("folds_quantity")
T = config.getParam("T")

# Loading tokenizer
tokenizer = load_tokenizer()

# Generating representation for each rep combination
for comb_no, comb in enumerate(REP_COMBS):
    print("____________________ Representing Comb No. %d / %d ____________________" % ((comb_no + 1), len(REP_COMBS)))

    # Collecting representations for every fold in rep combination 
    rep = []
    types = []
    for fold in comb:
        # Loading data (from comb model -> significant frags)
        df = pd.read_csv(DATA_PREDICTION_PATH % (model_id, fold.name + "." + comb_model_name(MODEL_NAME, FOLDS_QUANTITY)), sep=SEP)
        frag = df[PREDICTION_COLUMNS[3]]

        # Cutting out sequences types from ids
        rule = fold.cut_type_rule
        if rule != None:
            ids = df[PREDICTION_COLUMNS[0]]
            types.extend(ids.map(lambda id: id[:rule] if type(rule) == int else id.split(rule)[0]))

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
    x, y = tsne_2d(rep);

    # Plotting representation
    config_plot(ylim=False, size_scale=2, turn_of_default_style=True)
    plt.rc("axes", prop_cycle=cycler(color=COLORS_CYCLE))
    plt.axis("off")

    i = 0
    j = 0
    rep_name = ""
    for fold in comb:
        scope = fold.scope
        fold_name = fold.name

        if fold.cut_type_rule != None:
            # Printing each of fold sequences types separately
            fold_x = x[i:i+scope]
            fold_y = y[i:i+scope]
            fold_types = types[j:j+scope]

            # Sorting by sequences types
            zipped = zip(fold_x, fold_y, fold_types)
            zipped = sorted(zipped, key=lambda x: x[2])

            fx = []
            fy = []
            last_type = zipped[0][2]
            for x in zipped:
                current_type = x[2]

                if current_type != last_type:
                    plt.scatter(fx, fy, label=last_type, s=MARKER_SIZE)
                    fx = []
                    fy = []

                fx.append(x[0])
                fy.append(x[1])
                last_type = current_type
            plt.scatter(fx, fy, label=last_type, s=MARKER_SIZE)

            rep_name += TYPE_SEPARATION_LABEL + "."
            j += scope
        else:
            plt.scatter(x[i:i+scope], y[i:i+scope], label=fold_name, s=MARKER_SIZE)
        
        rep_name += fold_name + "."
        i += scope

    rep_name = rep_name[:-1]
    save_plot(REPRESENTATION_PATH % (model_id, rep_name))
