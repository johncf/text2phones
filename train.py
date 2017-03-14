#!/bin/python3

import tensorflow as tf
import numpy as np
import data
import model

batch_size = 10
input_max_length = 56
output_max_length = 60

reader = data.Reader('.', batch_size=batch_size, imax_len=input_max_length, omax_len=output_max_length)
input_ids, input_len, output_ids, output_len = reader.next_batch()
print([np.shape(tensor) for tensor in (input_ids, input_len, output_ids, output_len)])
print(input_ids)
print(output_ids)

m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
m.train(batch_size=batch_size, input_length=input_max_length, output_length=output_max_length)

#sess = tf.InteractiveSession()
#init = tf.global_variables_initializer()
#sess.run(init)
#
#res = sess.run(m.dec_out, feed_dict={m.input_data: input_ids, m.sequence_length: input_len})
#
#print(res.rnn_output.shape)
