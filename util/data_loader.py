import os
import math
import numpy as np

from Bio import SeqIO
from config import NEGATIVE_FOLD_PATH, POSITIVE_FOLD_PATH, NEGATIVE_DIR, POSITIVE_DIR


def load_trn_val_fold(fold_no):
    trn_marker = "trn%d" % fold_no
    val_marker = "val%d" % fold_no

    x_trn, y_trn = load_fold(NEGATIVE_FOLD_PATH % trn_marker, POSITIVE_FOLD_PATH % trn_marker)
    x_val, y_val = load_fold(NEGATIVE_FOLD_PATH % val_marker, POSITIVE_FOLD_PATH % val_marker)

    return x_trn, y_trn, x_val, y_val

def load_tst_fold(path):
    ids = []
    x = []

    for record in SeqIO.parse(path, "fasta"):
        ids.append(record.id)
        x.append(str(record.seq))

    return np.array(x), ids

def load_fold(negative_fold_path, positive_fold_path):
    x = []
    y = []

    # Loading negative data
    x_neg = []
    y_neg = []
    for record in SeqIO.parse(negative_fold_path, "fasta"):
        x_neg.append(str(record.seq))
        y_neg.append(0)

    # Loading positive data
    x_pos = []
    y_pos = []
    for record in SeqIO.parse(positive_fold_path, "fasta"):
        x_pos.append(str(record.seq))
        y_pos.append(1)

    # Merging negative with positive data (even distribution of positives in set) 
    x_neg_len = len(x_neg)
    scope = math.ceil(x_neg_len / len(x_pos)) + 1

    x_temp = []
    y_temp = []
    counter = 0
    for x_n, y_n in zip(x_neg, y_neg):
        x_temp.append(x_n)
        y_temp.append(y_n)
        counter += 1

        if counter == (scope - 1):
            x_temp.append(x_pos.pop())
            y_temp.append(y_pos.pop())

            # Shuffle
            shuffled_indexes = get_shuffled_indexes(len(x_temp))
            x_temp = np.array(x_temp)
            y_temp = np.array(y_temp)
            x_temp = x_temp[shuffled_indexes]
            y_temp = y_temp[shuffled_indexes]

            x.append(x_temp)
            y.append(y_temp)

            x_temp = []
            y_temp = []
            counter = 0

    x.append(x_temp)
    y.append(y_temp)

    return np.concatenate(x), np.concatenate(y)

def load_trn_seqs():
    trn_seqs = []

    def load_train_folds(folds_dir):
        for root, _, files in os.walk(folds_dir):
            for file in files:
                if "trn" in file:
                    for record in SeqIO.parse(os.path.join(root, file), "fasta"):
                        trn_seqs.append(str(record.seq))

    load_train_folds(NEGATIVE_DIR)
    load_train_folds(POSITIVE_DIR)

    return list(dict.fromkeys(trn_seqs)) # remove duplicates

def get_shuffled_indexes(length):
    indexes = np.arange(length)
    np.random.shuffle(indexes)
    return indexes
