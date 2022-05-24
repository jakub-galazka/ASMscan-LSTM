import random

from config import T, POPULATION, WEIGHTS


def random_pad_seqs(seqs):
    for i, seq in enumerate(seqs):
        k = T - len(seq)
        if k > 0:
            random_aa = random.choices(POPULATION, WEIGHTS, k=k)
            for aa in random_aa:
                seqs[i] += aa
