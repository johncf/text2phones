#!/bin/python3

import tensorflow as tf
import numpy as np
from data import Reader

reader = Reader('.', batch_size=4)
print(reader.next_batch())
