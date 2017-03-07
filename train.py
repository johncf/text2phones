#!/bin/python3

import tensorflow as tf
import numpy as np
import data

reader = data.Reader('.', imax_len=18)
input_ids, input_len, output_ids, output_len = reader.next_batch()
print([np.shape(tensor) for tensor in (input_ids, input_len, output_ids, output_len)])
print(input_ids)
print(output_ids)
