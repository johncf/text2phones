#!/bin/python3

import tensorflow as tf
import numpy as np
import data
import model

batch_size = 32
input_max_length = 56
output_max_length = 60

def main():
    reader = data.Reader('.', batch_size=batch_size,
                              imax_len=input_max_length,
                              omax_len=output_max_length)

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
    m.train(batch_size)
    #m.infer()

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    for i in range(1000):
        input_ids, input_len, output_ids, output_len = reader.next_batch()
        if input_ids is None:
            print("No more data!")
            break
        feed = { m.input_data: input_ids,
                 m.input_lengths: input_len,
                 m.output_data: output_ids,
                 m.output_lengths: output_len }
        if i % 50 == 0:
            train_accuracy = sess.run(m.accuracy, feed_dict=feed)
            print("step {0}, training accuracy {1}".format(i, train_accuracy))
        sess.run(m.train_step, feed_dict=feed)

main()
