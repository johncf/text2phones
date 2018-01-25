#!/bin/python3

import tensorflow as tf
import data
import model
from glob import glob
import sys
import time

batch_size = 100
input_max_length = 20
output_max_length = 16
learning_rate = 8e-4
checkpoint = "ckpts/model.ckpt"
logdir = "logdir/train"
start = 0

def main():
    reader = data.Reader('.', data='gist-data/data',
                              batch_size=batch_size,
                              in_maxlen=input_max_length,
                              out_maxlen=output_max_length)

    m = model.Model(input_size=reader.input_size, output_size=reader.output_size)
    m.train(batch_size, learning_rate, out_help=False, time_discount=0.08)

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

        prev_time = time.time()
        for i in range(start, 1000000):
            input_ids, input_len, output_ids, output_len = reader.next_batch()

            if input_ids is None:
                print("No more data!")
                break

            feed = { m.input_data: input_ids,
                     m.input_lengths: input_len,
                     m.output_data: output_ids,
                     m.output_lengths: output_len }
            summary_out, gstep, lrate, _ = sess.run([summaries, m.global_step, m.learning_rate, m.train_step], feed_dict=feed)

            if gstep % 100 == 0:
                summary_writer.add_summary(summary_out, gstep/50)
                accuracy, loss = sess.run([m.accuracy, m.losses], feed_dict=feed)
                sum_accuracy += accuracy
                count += 1
                cur_time = time.time()
                print("step {0}, rate {1:.6}, accuracy {2:.6}, loss {3:.4}, time {4:.3}".format(gstep, lrate, accuracy, loss, cur_time - prev_time))
                prev_time = cur_time

            if gstep % 2000 == 0:
                avg_accuracy = sum_accuracy/count
                print("Average accuracy:", avg_accuracy)
                sum_accuracy = 0.0
                count = 0

                summary_writer.flush()
                save_path = saver.save(sess, checkpoint)
                print("Model saved in file:", save_path)
                prev_time = time.time()

            if gstep % 500 == 0:
                sys.stdout.flush()
                prev_time = time.time()

        summary_writer.close()

main()
