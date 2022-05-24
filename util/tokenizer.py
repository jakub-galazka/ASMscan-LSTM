import os
import pickle
import tensorflow as tf

from config import TOKENIZER_PATH
from util.data_loader import load_trn_seqs


def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH): create_tokenizer()
    with open(TOKENIZER_PATH, 'rb') as handle:
        return pickle.load(handle)

def create_tokenizer():
    trn_seqs = load_trn_seqs()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(trn_seqs)

    with open (TOKENIZER_PATH, "wb") as handle:
        pickle.dump(tokenizer, handle, pickle.HIGHEST_PROTOCOL)
