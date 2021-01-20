import numpy as np


def multi_split_sequence(sequence, columns, window_size=36, output_size=36):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_in = i + window_size
        out_in = end_in + output_size
        if out_in < len(sequence):
            seq_x, seq_y = sequence[i:end_in], sequence[end_in:out_in, columns]
            x.append(seq_x)
            y.append(seq_y)

    return np.array(x), np.array(y)
