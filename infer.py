#!/bin/python3

import tensorflow as tf
import numpy as np
import data
import model
from glob import glob

checkpoint = "/tmp/model.ckpt"

def main():
    parser = data.Parser('.')

    m = model.Model(input_size=parser.input_size, output_size=parser.output_size)
    m.infer()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if len(glob(checkpoint + "*")) > 0:
            saver.restore(sess, checkpoint)
        else:
            print("No model found!")
            return

        ## -- debug --
        #np.set_printoptions(threshold=np.inf)
        #for v in tf.trainable_variables():
        #    print(v.name)
        #    print(sess.run(v))
        #    print()
        #return

        input_ = input('in> ')
        input_ids = parser.parse_input(input_)

        feed = { m.input_data: np.expand_dims(input_ids, 0) }
        output_ids = sess.run(m.output_ids, feed_dict=feed)

        print(parser.compose_output(output_ids[0]))

main()
