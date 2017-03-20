#!/bin/python3

import tensorflow as tf
import data
import model
from glob import glob
import sys

batch_size = 32
input_max_length = 30
output_max_length = 28
checkpoint = "ckpts/model.ckpt"

def main():
    reader = data.Reader('.', data='data-mix',
                              batch_size=batch_size,
                              in_maxlen=input_max_length,
                              out_maxlen=output_max_length)

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
    m.train(batch_size)

    init = tf.global_variables_initializer()

    avg_accuracy = 0.0
    count = 0

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if len(glob(checkpoint + "*")) > 0:
            saver.restore(sess, checkpoint)
            print("Model restored!")
        else:
            sess.run(init)
            print("Fresh variables!")

        for i in range(1000000):
            input_ids, input_len, output_ids, output_len = reader.next_batch()

            if input_ids is None:
                print("No more data!")
                break

            feed = { m.input_data: input_ids,
                     m.input_lengths: input_len,
                     m.output_data: output_ids,
                     m.output_lengths: output_len }
            sess.run(m.train_step, feed_dict=feed)

            if (i+1) % 50 == 0:
                train_accuracy = sess.run(m.accuracy, feed_dict=feed)
                avg_accuracy += train_accuracy
                count += 1
                print("step {0}, training accuracy {1}".format(i+1, train_accuracy))

            if (i+1) % 1000 == 0:
                print("Average accuracy:", avg_accuracy/count)
                avg_accuracy = 0.0
                count = 0

                save_path = saver.save(sess, checkpoint)
                print("Model saved in file:", save_path)

            if (i+1) % 500 == 0:
                sys.stdout.flush()

        writer = tf.summary.FileWriter('logdir', sess.graph)
        summaries = tf.summary.merge_all()
        summary_out = sess.run(summaries, feed_dict=feed)
        writer.add_summary(summary_out)
        writer.close()

main()
