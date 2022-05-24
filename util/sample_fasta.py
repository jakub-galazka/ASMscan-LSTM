import sys
import numpy as np

from Bio import SeqIO


path = sys.argv[1]
size = int(sys.argv[2])

raw_records = []
for record in SeqIO.parse(path, "fasta"):
    raw_records.append(record)

indexes = np.random.choice(np.arange(len(raw_records)), size, False)

sampled_records = []
for i in indexes:
    sampled_records.append(raw_records[i])

with open(path.split(".")[0] + "_sampled%d.fa" % size, "w") as f:
    SeqIO.write(sampled_records, f, "fasta")
