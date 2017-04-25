#!/bin/python3

import tensorflow as tf
import data
import model
from glob import glob
import sys

batch_size = 56
input_max_length = 28
output_max_length = 24
learning_rate = 4e-4
checkpoint = "ckpts/model101.ckpt"
logdir = "logdir/train101"
start = 0

def main():
    reader = data.Reader('.', data='data',
                              batch_size=batch_size,
                              in_maxlen=input_max_length,
                              out_maxlen=output_max_length)

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
    m.train(batch_size, learning_rate, out_help=False, time_discount=True, sampling_probability=0.2)

    summaries = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sum_accuracy = 0.0
    count = 0

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        if len(glob(checkpoint + "*")) > 0:
            saver.restore(sess, checkpoint)
            print("Model restored!")
        else:
            sess.run(init)
            print("Fresh variables!")

        for i in range(start, 1000000):
            input_ids, input_len, output_ids, output_len = reader.next_batch()

            if input_ids is None:
                print("No more data!")
                break

            feed = { m.input_data: input_ids,
                     m.input_lengths: input_len,
                     m.output_data: output_ids,
                     m.output_lengths: output_len }
            summary_out, _ = sess.run([summaries, m.train_step], feed_dict=feed)

            if (i+1) % 100 == 0:
                summary_writer.add_summary(summary_out, (i+1)/50)
                train_accuracy = sess.run(m.accuracy, feed_dict=feed)
                sum_accuracy += train_accuracy
                count += 1
                print("step {0}, training accuracy {1}".format(i+1, train_accuracy))

            if (i+1) % 2000 == 0:
                avg_accuracy = sum_accuracy/count
                print("Average accuracy:", avg_accuracy)
                sum_accuracy = 0.0
                count = 0

                summary_writer.flush()
                save_path = saver.save(sess, checkpoint)
                print("Model saved in file:", save_path)

            if (i+1) % 500 == 0:
                sys.stdout.flush()

        summary_writer.close()

main()
