import os
import numpy as np
from itertools import islice

class Reader:
    def __init__(self, root_path, data='data',
            isymbols='isymbols', osymbols='osymbols',
            imaxtime=64, omaxtime=64,
            batch_size=100, dtype=np.float32):
        self.data_handle = open(os.path.join(root_path, data))
        with open(os.path.join(root_path, isymbols)) as isyms:
            self.isymbols = dict([(sym.strip(), i) for i, sym in enumerate(isyms)])
        with open(os.path.join(root_path, osymbols)) as osyms:
            self.osymbols = dict([(sym.strip(), i) for i, sym in enumerate(osyms)])
        self.imaxtime = imaxtime
        self.omaxtime = omaxtime
        self.batch_size = batch_size
        self.dtype = dtype

    def _input_onehot(self, input_):
        input_ = input_.replace(' ', '_').lower()
        onehot = np.zeros((self.imaxtime, len(self.isymbols)))
        onehot[np.arange(len(input_)), np.array([self.isymbols[c] for c in input_])] = 1
        return onehot, len(input_)

    def _output_onehot(self, output_):
        output_ = output_.split()
        onehot = np.zeros((self.omaxtime, len(self.osymbols)))
        onehot[np.arange(len(output_)), np.array([self.osymbols[s] for s in output_])] = 1
        return onehot, len(output_)

    def next_batch(self):
        inputs, outputs = zip(*[line.strip().split(' :: ')
                                for line in islice(self.data_handle, self.batch_size)])
        inputs_onehot, inputs_len = zip(*[self._input_onehot(i) for i in inputs])
        outputs_onehot, outputs_len = zip(*[self._output_onehot(o) for o in outputs])
        return (np.array(inputs_onehot),
                np.array(inputs_len),
                np.array(outputs_onehot),
                np.array(outputs_len))
