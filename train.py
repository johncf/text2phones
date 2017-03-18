#!/bin/python3

import tensorflow as tf
import data
import model

batch_size = 32
input_max_length = 56
output_max_length = 60

def main():
    reader = data.Reader('.', data='data',
                              batch_size=batch_size,
                              in_maxlen=input_max_length,
                              out_maxlen=output_max_length)

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
    m.train(batch_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
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
            sess.run(m.train_step, feed_dict=feed)

            if i % 50 == 0:
                train_accuracy = sess.run(m.accuracy, feed_dict=feed)
                print("step {0}, training accuracy {1}".format(i, train_accuracy))

main()
