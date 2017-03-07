#!/bin/python3

import tensorflow as tf
import numpy as np
import data
import model

reader = data.Reader('.')
input_ids, input_len, output_ids, output_len = reader.next_batch()
print([np.shape(tensor) for tensor in (input_ids, input_len, output_ids, output_len)])
print(input_ids)
print(output_ids)

m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
m.train(batch_size=10, input_length=56, output_length=56)
