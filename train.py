#!/bin/python3

import tensorflow as tf
import numpy as np
import data
import model

batch_size = 10
input_max_length = 56
output_max_length = 60

def main():
    sess = tf.InteractiveSession()

    reader = data.Reader('.', batch_size=batch_size, imax_len=input_max_length, omax_len=output_max_length)
    input_ids, input_len, output_ids, output_len = reader.next_batch()

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size, output_maxlen=output_max_length)
    m.train(batch_size)
    #m.infer()

    init = tf.global_variables_initializer()
    sess.run(init)

    feed = { m.input_data: input_ids,
             m.input_lengths: input_len,
             m.output_data: output_ids,
             m.output_lengths: output_len }

    res = sess.run(m.losses, feed_dict=feed)

    print(output_ids[:, :11])
    print(res)

main()
