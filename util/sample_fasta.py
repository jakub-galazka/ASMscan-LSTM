import sys
import numpy as np

from Bio import SeqIO


# The script samples the sequences from the FASTA file
# Usage: python sample_fasta.py [path_to_fasta_file] [output_sample_set_size]

path = sys.argv[1]
size = int(sys.argv[2])

raw_records = []
for record in SeqIO.parse(path, "fasta"):
    raw_records.append(record)

if (size >= len(raw_records)):
    raise OverflowError("The number of samples cannot be greater than or equal to the size of the sampled collection!")

indexes = np.random.choice(np.arange(len(raw_records)), size, False)

sampled_records = []
for i in indexes:
    sampled_records.append(raw_records[i])

with open(path.split(".")[0] + "_sampled%d.fa" % size, "w") as f:
    SeqIO.write(sampled_records, f, "fasta")
